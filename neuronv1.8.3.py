import csv
import random
import time
import traceback
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


import modeltrainingprogram 

import pyqtgraph as pg
import networkx as nx
import numpy as np
from pyqtgraph.Qt import QtGui , QtCore  # Add this import
from pyqtgraph.Qt import QtWidgets


import pickle
import gzip
from collections import defaultdict


from scipy import stats
import json
import os
import mplcursors  # Fare etkileşimi için




class Neuron:
    next_id = 0  # Global olarak artan ID değeri
    
    def __init__(self, default_value:float=0.0, activation_type=None):
        self.value = default_value

        self.id = Neuron.next_id  # Otomatik ID ata
        Neuron.next_id += 1  # Sonraki nöron için ID artır
        self.activation_type = activation_type
        self.output = 0.0  # Çıktı değeri, aktivasyon fonksiyonundan sonra hesaplanacak
        self.weightedSum = 0

    def activation(self, x):
        if self.activation_type == 'sigmoid':
            x = np.clip(x, -500, 500)  # x'i -500 ile 500 arasında tut
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'tanh':
            return np.tanh(x)

        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def activation_derivative(self):
        if self.activation_type == 'sigmoid':
            # Çıktıyı 0.01-0.99 aralığına sıkıştır
            safe_output = np.clip(self.output, 0.01, 0.99)
            return safe_output * (1 - safe_output)  # f'(x) = f(x)(1 - f(x))
        elif self.activation_type == 'tanh':
            return 1 - self.output ** 2  # f'(x) = 1 - f(x)^2

        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def calculate_weighted_sum(self, layers, connections):
        weighted_sum = 0
        bias_sum = 0  # Bias toplamı
        
        for layer_idx in range(len(layers) - 1):
            for prev_neuron in layers[layer_idx]:
                for conn in connections[layer_idx].get(prev_neuron.id, []):
                    if conn.connectedTo[1] == self.id:
                        
                        weighted_sum += prev_neuron.value * conn.weight
                        bias_sum += conn.bias  # Bağlantı bias'larını topla
        
        self.weightedSum = weighted_sum + bias_sum  # Bias'ı ekle
        self.value = self.activation(self.weightedSum)
        self.output = self.value
        return self.value




class Connection:
    def __init__(self, weight=0, connectedToArg=[0, 0], bias=0.1):  # Varsayılan bias=0.1
        self.weight = weight
        self.connectedTo = connectedToArg
        self.bias = bias  # Bias parametresi eklendi
    
    def update_weight(self, learning_rate, delta):
        self.weight += learning_rate * delta
        self.bias += learning_rate * delta * 0.1  # Bias da güncelleniyor



visualizeNetwork =False
debug = True  # Global debug değişkeni
#cmd = "train_custom(veri.csv;2,5,2;0.0004)" #program başlar başlamaz çalışacak ilk komut
#cmd="train_custom(veri.csv;2,5,3,2;0.001;1;3)"
cmd="train_custom(parity_problem.csv;4,2,2;0.001;1;3)"

if debug:
    print("Debug değeri açık cmd komutları kaydedilecek.")
    cmdHistory=[]
    cmdHistory.append(cmd)


# Ağ oluşturma
randomMinWeight = -2.0
randomMaxWeight = 2.0



activation_types = ['sigmoid', 'tanh']
defaultNeuronActivationType='tanh'






# Önce boş bir layers listesi oluştur
layers = []



# Bağlantıları oluşturma

connections = {}
def setConnections(preserve_weights=True):
    global layers, connections
    
    # Eski ağırlıkları sakla
    old_weights = {}
    if preserve_weights:
        for layer_idx in connections:
            for neuron_id in connections[layer_idx]:
                for conn in connections[layer_idx][neuron_id]:
                    key = (layer_idx, neuron_id, conn.connectedTo[1])
                    old_weights[key] = conn.weight
    
    # Yeni bağlantıları oluştur
    new_connections = {layer_idx: {} for layer_idx in range(len(layers) - 1)}
    for layer_idx in range(len(layers) - 1):
        for neuron in layers[layer_idx]:
            for next_neuron in layers[layer_idx + 1]:
                key = (layer_idx, neuron.id, next_neuron.id)
                
                if preserve_weights and key in old_weights:
                    # Eski ağırlığı koru
                    weight = old_weights[key]
                else:
                    # Yeni ağırlık oluştur
                    weight = random.uniform(-1/np.sqrt(len(layers[layer_idx])), 
                                    1/np.sqrt(len(layers[layer_idx])))
                
                conn = Connection(connectedToArg=[neuron.id, next_neuron.id], weight=weight)
                
                if neuron.id not in new_connections[layer_idx]:
                    new_connections[layer_idx][neuron.id] = []
                new_connections[layer_idx][neuron.id].append(conn)
    
    connections = new_connections

def setLayers(neuronInLayers):
    """Katmanları ve nöron sayılarını ayarlar"""
    global layers  # Global layers listesini kullanacağımızı belirtiyoruz
    layers.clear()  # Önceki katmanları temizle
    
    for layerIndex,neuronCount in enumerate(neuronInLayers):

        # Her katman için yeni nöron listesi oluştur
        layer = [Neuron(default_value=1,activation_type= "sigmoid" if layerIndex == len(neuronInLayers) -1 else "tanh") for _ in range(neuronCount)]
        layers.append(layer)
    
    setConnections(preserve_weights=False)


def createFileName():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"trainingDatas/network_{timestamp}.pkl.gz"





def save_network_optimized(filename=None):
    if filename is None:
        filename = createFileName()
    """Yüksek performanslı binary kayıt fonksiyonu"""
    network_data = {
        'layers': [
            [
                {
                    'id': n.id,
                    'value': n.value,
                    'output': n.output,
                    'weightedSum': n.weightedSum,
                    'activation_type': n.activation_type
                } 
                for n in layer
            ] 
            for layer in layers
        ],
        'connections': [
            (layer_idx, conn.connectedTo[0], conn.connectedTo[1], conn.weight, conn.bias)
            for layer_idx in connections
            for neuron_id in connections[layer_idx]
            for conn in connections[layer_idx][neuron_id]
        ],
        'config': (
            randomMinWeight,
            randomMaxWeight,
            defaultNeuronActivationType,
            visualizeNetwork,
            debug
        ),
        'next_id': Neuron.next_id
    }
    
    with gzip.open(filename, 'wb') as f:
        pickle.dump(network_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_network_optimized(filename):
    """Yüksek performanslı binary yükleme fonksiyonu"""
    global layers, connections, Neuron
    
    with gzip.open(filename, 'rb') as f:
        network_data = pickle.load(f)
    
    # Global değişkenleri güncelle
    (randomMinWeight, randomMaxWeight, defaultNeuronActivationType, 
     visualizeNetwork, debug) = network_data['config']
    Neuron.next_id = network_data['next_id']
    
    # Katmanları yeniden oluştur
    layers = []
    for layer_data in network_data['layers']:
        layer = []
        for n_data in layer_data:
            n = Neuron(activation_type=n_data['activation_type'])
            n.id = n_data['id']
            n.value = n_data['value']
            n.output = n_data['output']
            n.weightedSum = n_data['weightedSum']
            layer.append(n)
        layers.append(layer)
    
    # Bağlantıları yeniden oluştur (defaultdict ile hızlı erişim)
    connections = defaultdict(dict)
    for conn_data in network_data['connections']:
        layer_idx, from_id, to_id, weight, bias = conn_data
        if from_id not in connections[layer_idx]:
            connections[layer_idx][from_id] = []
        connections[layer_idx][from_id].append(
            Connection(weight=weight, connectedToArg=[from_id, to_id], bias=bias)
        )
    
    return True







def scale_value(x, source_min, source_max, target_min, target_max):
    """
    Bir değeri kaynak aralıktan hedef aralığa ölçeklendirir.

    :param x: Dönüştürülecek değer
    :param source_min: Kaynak aralığın alt sınırı
    :param source_max: Kaynak aralığın üst sınırı
    :param target_min: Hedef aralığın alt sınırı
    :param target_max: Hedef aralığın üst sınırı
    :return: Ölçeklendirilmiş değer
    """
    return target_min + ((x - source_min) / (source_max - source_min)) * (target_max - target_min)


def runAI():
    for layer in layers[1:]:
        for neuron in layer:
            #print(f"Nöron {neuron.id}: {neuron.value}")
            neuron.calculate_weighted_sum(layers,connections)
    #print(f"Son değer: {scale_value(get_neuron_by_id(30).value,0,1,0,8)}")
    lastNeuronValues =[]
    for neuron in layers[-1]:
        lastNeuronValues.append(neuron.value)
    return lastNeuronValues



# Global değişkenler
global win, plot, scatter, lines, app
win = None
plot = None
scatter = None
lines = []
app = None





def visualize_network(layers, connections, node_size=20,refresh=False):
    if not visualizeNetwork:
        return
    global win, plot, scatter, lines, app

    if win is None or not refresh:
        if QtWidgets.QApplication.instance() is None:
            app = QtWidgets.QApplication([])
        else:
            app = QtWidgets.QApplication.instance()
        
        win = pg.GraphicsLayoutWidget(show=True,size=(1200,800))
        win.setWindowTitle("Sinir Ağı Görselleştirme")
        plot = win.addPlot()
        #view = plot.getViewBox()
        plot.hideAxis('bottom')
        plot.hideAxis('left')
        #view.setAspectLocked(True)
        win.setBackground('#f0f0f0')
    else:
        plot.clear()
        for line in lines:
            plot.removeItem(line)
        lines.clear()
        if scatter:
            plot.removeItem(scatter)
    #app = QtWidgets.QApplication([])
    #win = pg.GraphicsLayoutWidget(show=True, size=(1200, 800))
    #win.setWindowTitle("Sinir Ağı Görselleştirme (Scatter + Lines Hybrid)")
    
    
    #plot = win.addPlot()

    view = plot.getViewBox()
    view.setMouseEnabled(x=True, y=True)  # Yakınlaştırma ve kaydırma aktif
    view.setAspectLocked(False)  # Oran kilidi kapalı (serbest zoom)
    view.setMenuEnabled(False)  # Sağ tık menüsü kapalı
    view.wheelZoomFactor = 1.1  # Zoom hassasiyeti

    # 1. Nöron pozisyonları ve renkleri
    pos = {}
    node_colors = []
    node_ids = []
    node_values = []
    
    for layer_idx, layer in enumerate(layers):
        layer_x = layer_idx * 3.0
        for neuron_idx, neuron in enumerate(layer):
            y_pos = -(neuron_idx - len(layer)/2) * 1.5
            pos[neuron.id] = (layer_x, y_pos)
            node_ids.append(neuron.id)
            node_values.append(neuron.value)
            
            # Aktivasyona göre renk (kırmızı: düşük, yeşil: yüksek)
            norm_val = max(0, min(1, neuron.value))  # 0-1 arasına sıkıştır
            node_colors.append(pg.mkColor(
                int(255 * (1 - norm_val)),  # R
                int(255 * norm_val),        # G
                0,                         # B
                200                        # Alpha
            ))
    


    # 2. Bağlantı verilerini topla
    edges = []
    edge_weights = []
    edge_info = []
    
    for layer_idx, layer_conns in connections.items():
        for src_id, conn_list in layer_conns.items():
            for conn in conn_list:
                if conn.connectedTo[0] in pos and conn.connectedTo[1] in pos:
                    edges.append((conn.connectedTo[0], conn.connectedTo[1]))
                    edge_weights.append(conn.weight)
                    edge_info.append({
                        'from': conn.connectedTo[0],
                        'to': conn.connectedTo[1],
                        'weight': conn.weight
                    })
                else:
                    print(f"Uyarı: Geçersiz bağlantı {conn.connectedTo}")

    # 3. Nöronları çiz (ScatterPlot)
    scatter = pg.ScatterPlotItem(
        pos=np.array(list(pos.values())),
        size=node_size,
        brush=node_colors,
        pen=pg.mkPen('k', width=1),
        pxMode=True
    )
    plot.addItem(scatter)

    # 4. Bağlantıları çiz (LineCollection benzeri yaklaşım)
    if edges and edge_weights:
        # Ağırlıkları normalize et
        min_w, max_w = min(edge_weights), max(edge_weights)
        range_w = max_w - min_w if max_w != min_w else 1
        
        # Her bağlantı için ayrı çizgi oluştur
        for i, (u, v) in enumerate(edges):
            # Ağırlığa göre stil belirle
            norm_w = (edge_weights[i] - min_w) / range_w
            
            if norm_w < 0.5:  # Negatif ağırlık (mavi)
                intensity = int(255 * (1 - 2*norm_w))
                color = pg.mkColor(0, 0, 255, intensity)
                width = 1 + 3 * (1 - 2*norm_w)
            else:  # Pozitif ağırlık (kırmızı)
                intensity = int(255 * (2*(norm_w - 0.5)))
                color = pg.mkColor(255, 0, 0, intensity)
                width = 1 + 3 * (2*(norm_w - 0.5))
            
            # Çizgiyi ekle
            line = pg.PlotDataItem(
                x=[pos[u][0], pos[v][0]],
                y=[pos[u][1], pos[v][1]],
                pen=pg.mkPen(color, width=width)
            )
            plot.addItem(line)

    # 5. Nöron etiketleri
    for neuron_id, (x, y) in pos.items():
        text = pg.TextItem(str(neuron_id), color='w', anchor=(0.5, 0.5))
        text.setFont(QtGui.QFont('Arial', max(8, node_size//3)))
        text.setPos(x, y)
        view.addItem(text)

    # 6. Tooltip işlevselliği
    tooltip = pg.TextItem(color='k', anchor=(0, 1), fill=(255, 255, 255, 200))
    edge_tooltip = pg.TextItem(color='k', anchor=(0.5, 0.5), fill=(255, 255, 255, 200))
    view.addItem(tooltip)
    view.addItem(edge_tooltip)
    tooltip.hide()
    edge_tooltip.hide()

    def on_hover(event):
        mouse_pos = view.mapSceneToView(event)
        
        # Önce bağlantıları kontrol et
        closest_edge = None
        min_edge_dist = float('inf')
        
        for i, (u, v) in enumerate(edges):
            start = np.array(pos[u])
            end = np.array(pos[v])
            line_vec = end - start
            mouse_vec = np.array([mouse_pos.x(), mouse_pos.y()]) - start
            
            t = np.dot(mouse_vec, line_vec) / np.dot(line_vec, line_vec)
            t = max(0, min(1, t))
            projection = start + t * line_vec
            dist = np.linalg.norm(np.array([mouse_pos.x(), mouse_pos.y()]) - projection)
            
            if dist < min_edge_dist:
                min_edge_dist = dist
                closest_edge = i
        
        if closest_edge is not None and min_edge_dist < 0.15:
            u, v = edges[closest_edge]
            weight = edge_weights[closest_edge]
            edge_tooltip.setText(f"Bağlantı: {u} → {v}\nAğırlık: {weight:.6f}")
            edge_pos = (np.array(pos[u]) + np.array(pos[v])) / 2
            edge_tooltip.setPos(edge_pos[0], edge_pos[1])
            edge_tooltip.show()
            tooltip.hide()
        else:
            edge_tooltip.hide()
            
            # Sonra nöronları kontrol et
            closest_node = None
            min_dist = float('inf')
            
            for neuron_id, (x, y) in pos.items():
                dist = ((mouse_pos.x() - x)**2 + (mouse_pos.y() - y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_node = neuron_id
            
            if min_dist < (node_size/20) and closest_node is not None:
                neuron = get_neuron_by_id(closest_node)
                incoming = [e for e in edge_info if e['to'] == closest_node]
                outgoing = [e for e in edge_info if e['from'] == closest_node]
                
                text = (f"Nöron {closest_node}\n"
                       f"Değer: {neuron.value:.4f}\n"
                       f"Toplam: {neuron.weightedSum:.4f}\n"
                       f"\nGelen bağlantılar ({len(incoming)}):\n")
                
                text += "\n".join(f"  {conn['from']} → {conn['weight']:.4f}" for conn in incoming[:5])  # İlk 5
                if len(incoming) > 5: text += "\n  ..."
                
                text += f"\n\nGiden bağlantılar ({len(outgoing)}):\n"
                text += "\n".join(f"  → {conn['to']} ({conn['weight']:.4f})" for conn in outgoing[:5])
                if len(outgoing) > 5: text += "\n  ..."
                
                tooltip.setText(text)
                tooltip.setPos(mouse_pos.x(), mouse_pos.y())
                tooltip.show()
            else:
                tooltip.hide()
    
    view.scene().sigMouseMoved.connect(on_hover)

    # 7. Gösterge (legend)
    legend = pg.LegendItem(offset=(10, 10), size=(150, 100))
    legend.setParentItem(view)
    legend.addItem(pg.PlotDataItem(pen=pg.mkPen((255,0,0), width=3)), "Pozitif ağırlık")
    legend.addItem(pg.PlotDataItem(pen=pg.mkPen((0,0,255), width=3)), "Negatif ağırlık")
    legend.addItem(pg.PlotDataItem(brush=pg.mkBrush((255,100,100))), "Düşük aktivasyon")
    legend.addItem(pg.PlotDataItem(brush=pg.mkBrush((100,255,100))), "Yüksek aktivasyon")

    # 8. Performans ayarları
    plot.setMenuEnabled(False)
    
    if not refresh:
        win.show()
        app.exec_()
    else:
        win.show()
        QtWidgets.QApplication.processEvents()






def change_weight(connections, from_id, to_id, new_weight):
    """
    Belirli bir bağlantının ağırlığını değiştirir.

    :param connections: Katmanlar arası bağlantılar
    :param from_id: Bağlantıdan gelen nöronun ID'si
    :param to_id: Bağlantıya giden nöronun ID'si
    :param new_weight: Yeni ağırlık
    """
    # connections dict'si üzerinden gezerek doğru bağlantıyı bulalım
    for layer_connections in connections.values():
        for neuron_id, conn_list in layer_connections.items():
            for conn in conn_list:
                # Bağlantı [from_id, to_id] olup olmadığını kontrol et
                if conn.connectedTo == [from_id, to_id]:
                    conn.weight = new_weight  # Yeni ağırlığı güncelle
                    print(f"Bağlantı güncellendi: {from_id} -> {to_id} yeni weight: {new_weight}")
                    return  # İşlem tamamlandığında fonksiyonu sonlandır

    print(f"Hata: {from_id} ile {to_id} arasında bağlantı bulunamadı.")  # Bağlantı bulunamazsa hata mesajı














def get_neuron_by_id(neuron_id, layersArg=layers):
    for layer in layersArg:
        for neuron in layer:
            if neuron.id == neuron_id:
                return neuron
    return None  # Eğer nöron bulunamazsa None döndür




# Hata payı fonksiyonu
def hata_payi(target, output):
    # Listeleri numpy dizilerine dönüştür
    target = np.array(target)
    output = np.array(output)
    return np.mean((target - output) ** 2)






def clearGUI():
    global win
    if win is not None:
        win.close()
        win = None


def get_connections(layer_idx=None, detailed=False):
    """
    Ağdaki bağlantı bilgilerini döndürür.
    
    Parametreler:
    - layer_idx: Belirli bir katmanın bağlantılarını getir (None ise tüm katmanlar)
    - detailed: Detaylı bilgi (kimden kime) ekler
    
    Dönüş Değeri:
    - Eğer detailed=False: {layer_idx: {from_id: [weight1, weight2, ...]}}
    - Eğer detailed=True: {layer_idx: [(from_id, to_id, weight), ...]}
    """
    result = {}
    
    # Tüm katmanlar için
    if layer_idx is None:
        target_layers = connections.keys()
    else:
        if layer_idx not in connections:
            print(f"Uyarı: {layer_idx}. katman bulunamadı!")
            return {}
        target_layers = [layer_idx]
    
    for l_idx in target_layers:
        layer_conns = connections[l_idx]
        
        if not detailed:
            # Basit format: {from_id: [weight1, weight2, ...]}
            simple_format = {}
            for from_id, conn_list in layer_conns.items():
                simple_format[from_id] = [conn.weight for conn in conn_list]
            result[l_idx] = simple_format
        else:
            # Detaylı format: [(from_id, to_id, weight), ...]
            detailed_format = []
            for from_id, conn_list in layer_conns.items():
                for conn in conn_list:
                    detailed_format.append((
                        from_id,
                        conn.connectedTo[1],
                        conn.weight
                    ))
            result[l_idx] = detailed_format
    
    return result
"""
# KULLANIM ÖRNEKLERİ:

# 1. Tüm bağlantıları basit şekilde alma
all_connections = get_connections()
print("Tüm bağlantılar (basit):", all_connections)

# 2. Belirli bir katmanın bağlantılarını detaylı alma
layer_1_conns = get_connections(layer_idx=1, detailed=True)
print("\n1. katman bağlantıları (detaylı):")
for conn in layer_1_conns[1]:
    print(f"{conn[0]} -> {conn[1]} | Ağırlık: {conn[2]:.4f}")"""

# 3. Belirli bir nöronun bağlantılarını bulma
def get_neuron_connections(neuron_id, incoming=True, outgoing=True):
    """
    Returns list of Connection objects instead of tuples
    """
    found = []
    
    # Önce nöronun hangi katmanda olduğunu bulalım
    neuron_layer = None
    for layer_idx, layer in enumerate(layers):
        for neuron in layer:
            if neuron.id == neuron_id:
                neuron_layer = layer_idx
                break
        if neuron_layer is not None:
            break
    
    if neuron_layer is None:
        print(f"Uyarı: {neuron_id} ID'li nöron bulunamadı!")
        return found
    
    # Gelen bağlantılar (önceki katmandan)
    if incoming and neuron_layer > 0:
        prev_layer_idx = neuron_layer - 1
        if prev_layer_idx in connections:
            for from_id, conn_list in connections[prev_layer_idx].items():
                for conn in conn_list:
                    if conn.connectedTo[1] == neuron_id:
                        found.append(conn)  # Return the Connection object itself
    
    # Giden bağlantılar (sonraki katmana)
    if outgoing and neuron_layer < len(layers) - 1:
        current_layer_idx = neuron_layer
        if current_layer_idx in connections and neuron_id in connections[current_layer_idx]:
            for conn in connections[current_layer_idx][neuron_id]:
                found.append(conn)  # Return the Connection object itself
    
    return found






def print_error_progress(current_error, target_error=0.01, width=50):
    """Hata değerine göre ilerleme çubuğu"""
    if target_error is None or target_error <= 0:
        target_error = 0.01  # Varsayılan değer
    
    progress = min(1.0, max(0.0, 1 - (current_error / target_error)))
    filled = int(progress * width)
    bar = '[' + '=' * filled + ' ' * (width - filled) + ']'
    print(f"\rHata Modu: {current_error:.6f} {bar} %{progress*100:.1f}", end='')
    if progress >= 0.99:
        print(f"\nHedef hata değerine ulaşıldı: {current_error:.6f} <= {target_error:.6f}")

def print_epoch_progress(current_epoch, total_epochs, current_error, width=50):
    """Epoch sayısına göre ilerleme çubuğu"""
    progress = (current_epoch / total_epochs) if total_epochs > 0 else 0
    filled = int(progress * width)
    bar = '[' + '=' * filled + ' ' * (width - filled) + ']'
    print(f"\rEpoch Modu: {current_epoch}/{total_epochs} {bar} %{progress*100:.1f} | Hata: {current_error:.6f}", end='')


def compute_gradients(y_true, output):
    # First collect all neurons from all layers
    all_neurons = []
    for layer in layers:
        all_neurons.extend(layer)
    
    gradients = {neuron.id: 0.0 for neuron in all_neurons}  # Create dictionary for all neurons
    
    # Output layer gradients
    for i, neuron in enumerate(layers[-1]):
        error = y_true[i] - output[i] if i < len(y_true) else 0
        gradients[neuron.id] = error * neuron.activation_derivative()
    
    # Hidden layers gradients (backwards)
    for layer in reversed(layers[:-1]):
        for neuron in layer:
            # Sum of gradients from next layer weighted by connection weights
            grad_sum = 0
            for conn in get_neuron_connections(neuron.id, outgoing=True):
                grad_sum += conn.weight * gradients[conn.connectedTo[1]]
            
            gradients[neuron.id] = grad_sum * neuron.activation_derivative()
    
    return gradients



import signal
import json
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Global değişkenler
error_history = []
epoch_history = []
learning_rate_history=[]
start_time = None
enable_logging = True  # Loglama varsayılan olarak kapalı

stopEpoch = False #ctrl C yapınca eğitimi durdurması için


def signal_handler(sig, frame):
    """Ctrl+C ile çıkış yakalandığında çağrılacak fonksiyon"""
    global enable_logging,stopEpoch
    
    print("\nEğitim durduruldu.")
    stopEpoch =True
    if enable_logging:
        print("Veriler kaydediliyor...")
        
        #visualize_saved_errors(save_and_plot_errors())

    

import os

def klasor_hazirla(yol):
    """Verilen yol için klasör yapısını hazırlar"""
    try:
        os.makedirs(yol, exist_ok=True)
        print(f"Klasör yapısı hazır: {yol}")
        return True
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return False


def save_and_plot_errors():
    """Hata geçmişini kaydet ve görselleştir"""
    global error_history, epoch_history, start_time
    
    if not error_history:
        print("Kaydedilecek veri yok.")
        return
    
    # Çıktı dosyasının adını belirle - sadece bir dosya oluştur
    outputFolder="trainingDatas/"
    klasor_hazirla(outputFolder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_base = f"training_errors_{timestamp}"
    output_file_json = f"{output_file_base}.json"
    output_file_png = f"{output_file_base}.png"
    
    # Toplam eğitim süresini hesapla
    total_time = time.time() - start_time if start_time else 0
    
    # Veriyi JSON formatında kaydet
    data = {
        "errors": error_history,
        "epochs": epoch_history,
        "total_time_seconds": total_time,
        "final_error": error_history[-1] if error_history else None,
        "learning_rates": learning_rate_history  # Bu yeni eklenen kısım
    }
    
    with open(outputFolder+output_file_json, 'w') as f:
        json.dump(data, f)
    
    print(f"Hata verileri {output_file_json} dosyasına kaydedildi.")
    
    
    
    
    return outputFolder+output_file_json

def train_network(X_train, y_train, batch_size=1, epochs=None, intelligenceValue=None, learning_rate=0.05, output_graph=enable_logging):
    global error_history, epoch_history, start_time, enable_logging, learning_rate_history
    
    if enable_logging:
        print("Hata grafiği kaydı etkin. Eğitim sonunda grafik oluşturulacak.")
    
    error_history = []
    epoch_history = []
    learning_rate_history = []
    newLR = 0.0
    
    signal.signal(signal.SIGINT, signal_handler)
    
    cortical_column = CorticalColumn(learning_rateArg=learning_rate, targetError=epochs if epochs <1 else None,
                                     maxEpochForTargetError=len(X_train)*50,
                                     originalNetworkModel=[len(liste) for liste in layers])

    
    avg_error = float('inf')
    epoch = 0
    total_samples = len(X_train)
    start_time = time.time()

    if len(layers[0]) != len(X_train[0]):
        print(f"Uyarı: Giriş boyutu uyumsuz! Ağ girişi: {len(layers[0])}, Veri girişi: {len(X_train[0])}")
        return

    try:
        while True:
            cortical_column.current_epoch = epoch
            total_error = 0
            processed_samples = 0
            epoch_gradients = []
            korteksChanges = []
            
            newLR,_ = cortical_column.monitor_network(avg_error)
            if _ is not None and epochs < 1 :
                epochs = _
                

            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                batch_error = 0
                
                for X, y in zip(X_batch, y_batch):
                    cortical_column.backpropagation(X, y)
                    
                    output = [neuron.value for neuron in layers[-1][:len(y)]]
                    error = hata_payi(y, output)
                    batch_error += error
                
                batch_error /= len(X_batch)
                total_error += batch_error * len(X_batch)
                processed_samples += len(X_batch)
                avg_error = total_error / processed_samples
                
                # SADECE BATCH SONUNDA LOGLAMA
                elapsed_time = time.time() - start_time
                samples_per_sec = processed_samples / elapsed_time if elapsed_time > 0 else 0
                remaining_time = (total_samples - processed_samples) / samples_per_sec if samples_per_sec > 0 else 0
                
                print(f"\nEpoch {epoch+1}/{epochs if epochs is not None else '∞'} - İlerleme: {processed_samples}/{total_samples} ({100*processed_samples/total_samples:.1f}%)")
                print(f"Ortalama Hata: {avg_error:.6f}")
                print(f"Geçen Süre: {elapsed_time/60:.1f} dak - Tahmini Kalan Süre: {remaining_time/60:.1f} dak")
                print(f"Örnek/Saniye: {samples_per_sec:.1f}")



            if enable_logging:
                    error_history.append(avg_error)
                    epoch_history.append(epoch + processed_samples/total_samples)
                    learning_rate_history.append(newLR)   
                
            
            if ((epochs > 1 and epoch >= epochs) or (epochs <1 and epochs>avg_error)) or stopEpoch == True:
                if epoch % 50 == 0 and debug:
                    cortical_column.log_change('epoch_summary', {
                        'average_error': avg_error,
                        'batch_progress': processed_samples/total_samples
                    })
                break
            
            epoch += 1
        
        total_time = time.time() - start_time
        print(f"\n=== EĞİTİM TAMAMLANDI ===")
        print(f"Toplam Süre: {total_time/60:.1f} dakika | Toplam saniye: {total_time:.3f}")
        print(f"Son Hata: {avg_error:.6f}")
        print(f"Toplam Epoch: {epoch}")
        print(f"Final Ağ Yapısı: {[len(layer) for layer in layers]}")

        try:
            filename = createFileName()
            save_network_optimized(filename)
            print(f"Ağ yapısı {filename} dosyasına kaydedildi")
        except Exception as a:
            print("Modeli dosyaya kaydetme sırasında hata meydana geldi.")
            traceback.print_exc()

            

        if enable_logging:
            visualize_saved_errors(save_and_plot_errors())
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if debug:
            cortical_column.log_change('training_error', {
                'error_type': str(type(e)),
                'message': str(e),
                'last_epoch': epoch
            })
        raise
    
    return cortical_column, avg_error



def visualize_saved_errors(filename, last_20Arg=0.8):
    """Kaydedilmiş hata verilerini gelişmiş grafiklerle görselleştir"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    errors = np.array(data["errors"])
    epochs = np.array(data["epochs"])
    learning_rates = np.array(data.get("learning_rates", [0.05]*len(epochs)))
    
    # Ana grafik
    plt.figure(figsize=(15, 10))
    
    # 1. Hata eğrisi (ana grafik)
    ax1 = plt.subplot(2, 2, (1, 3))  # 2 satır, 2 sütun, 1 ve 3'ü birleştir
    main_plot = plt.plot(epochs, errors, 'b-', linewidth=1, label='Ortalama Hata')
    scatter1 = plt.plot(epochs, errors, 'ro', markersize=1)[0]
    
    # Eğilim çizgisi ekleme
    z = np.polyfit(epochs, errors, 3)
    p = np.poly1d(z)
    trend_line = plt.plot(epochs, p(epochs), "r--", linewidth=2, label='Eğilim Çizgisi')[0]
    
    # Dönüm noktalarını bulma
    diff = np.diff(errors)
    turning_points = np.where(np.diff(np.sign(diff)))[0] + 1
    
    if len(turning_points) > 0:
        for tp in turning_points:
            plt.plot(epochs[tp], errors[tp], 'go', markersize=2, label='Dönüm Noktası' if tp == turning_points[0] else "")
    
    # Grafik özelleştirme
    plt.title('Eğitim Sırasında Hata Değişimi ve Eğilimi')
    plt.xlabel('Epoch')
    plt.ylabel('Ortalama Hata')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Son 20% epoch için yakınlaştırılmış grafik
    ax2 = plt.subplot(2, 2, 2)
    last_20 = int(len(epochs) * last_20Arg)
    line2 = plt.plot(epochs[last_20:], errors[last_20:], 'b-', linewidth=1.5)[0]
    scatter2 = plt.plot(epochs[last_20:], errors[last_20:], 'ro', markersize=2)[0]
    
    # Son bölüm için lineer regresyon
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        epochs[last_20:], errors[last_20:])
    reg_line = plt.plot(epochs[last_20:], intercept + slope*epochs[last_20:], 
             'g--', linewidth=2, 
             label=f'Eğim: {slope:.2e}\nR²: {r_value**2:.2f}')[0]
    
    plt.title(f'Son %{int(100-last_20Arg*100)} Epoch Yakınlaştırma')
    plt.xlabel('Epoch')
    plt.ylabel('Hata')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Learning Rate Değişimi Grafiği
    ax3 = plt.subplot(2, 2, 4)
    lr_line = plt.plot(epochs, learning_rates, 'm-', linewidth=1.5, label='Learning Rate')[0]
    lr_scatter = plt.plot(epochs, learning_rates, 'co', markersize=2)[0]
    
    # Learning rate için eğilim çizgisi
    z_lr = np.polyfit(epochs, learning_rates, 1)
    p_lr = np.poly1d(z_lr)
    plt.plot(epochs, p_lr(epochs), "k--", linewidth=1, label=f'Eğilim: {z_lr[0]:.2e}x + {z_lr[1]:.2f}')
    
    plt.title('Learning Rate Değişimi')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Genel bilgiler
    stats_text = (
        f"Başlangıç Hata: {errors[0]:.6f}\n"
        f"Son Hata: {errors[-1]:.6f}\n"
        f"En Düşük Hata: {np.min(errors):.6f}\n"
        f"Ortalama Hata: {np.mean(errors):.6f}\n"
        f"Standart Sapma: {np.std(errors):.6f}\n"
        f"Dönüm Noktaları: {len(turning_points)}\n"
        f"Başlangıç LR: {learning_rates[0]:.6f}\n"
        f"Son LR: {learning_rates[-1]:.6f}\n"
        f"Toplam Epoch: {len(epochs)}\n"
        f"Toplam Süre: {data['total_time_seconds']:.2f} sn"
    )
    
    plt.figtext(0.75, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.5), 
                fontsize=9)
    
    plt.tight_layout()
    
    # Grafiği kaydet
    output_file = os.path.splitext(filename)[0] + "_advanced_viz.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Gelişmiş hata grafiği {output_file} dosyasına kaydedildi.")
    
    # Eğilim analizi
    analyze_trend(errors, epochs, last_20Arg=last_20Arg)
    
    # Fare etkileşimi ekleme
    def format_annotation(sel):
        x, y = sel.target
        epoch = int(x)
        if sel.artist in [scatter1, scatter2]:  # Hata grafiklerindeki noktalar
            error = y
            sel.annotation.set_text(f"Epoch: {epoch}\nHata: {error:.6f}")
        elif sel.artist == lr_scatter:  # Learning rate grafiğindeki noktalar
            lr = y
            sel.annotation.set_text(f"Epoch: {epoch}\nLearning Rate: {lr:.6f}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
    
    # Tüm grafikler için cursor ekle
    crs1 = mplcursors.cursor([scatter1, scatter2, lr_scatter], hover=True)
    crs1.connect("add", format_annotation)
    if debug:
        plt.show(block=True)
    else:
        plt.show()

def analyze_trend(errors, epochs,last_20Arg):
    """Hata eğilimini analiz eder ve yorumlar"""
    # Son %(default 20)'lik kısım için eğim analizi

    last_20 = int(len(epochs) * last_20Arg)
    slope, _, _, _, _ = stats.linregress(epochs[last_20:], errors[last_20:])
    
    print("\n=== HATA EĞİLİM ANALİZİ ===")
    print(f"Son hata değeri: {errors[-1]:.6f}")
    print(f"Son %{100-last_20Arg*100} epoch'taki ortalama hata eğimi: {slope:.2e}")
    
    if slope > 1e-6:
        print("UYARI: Hatalarda artış eğilimi var! Model overfitting olabilir veya öğrenme oranı yüksek olabilir.")
    elif slope < -1e-6:
        print("Hatalarda düşüş eğilimi devam ediyor. Eğitime devam edilebilir.")
    else:
        print("Hatalar sabitlenmiş görünüyor. Daha fazla eğitimin faydası olmayabilir.")
    
    # Yakınsama kontrolü
    last_10_errors = errors[-10:]
    std_last_10 = np.std(last_10_errors)
    if std_last_10 < 0.001:
        print(f"Hatalar yakınsamış (son 10 epoch std: {std_last_10:.6f})")
    else:
        print(f"Hatalar henüz tam yakınsamadı (son 10 epoch std: {std_last_10:.6f})")
    
    # Öneriler
    print("\n=== ÖNERİLER ===")
    if slope > 0 and len(errors) > 50:
        print("- Öğrenme oranını azaltmayı deneyin")
        print("- Regularization ekleyin")
        print("- Early stopping uygulayın")
    elif slope < -1e-4:
        print("- Model hala öğreniyor, eğitime devam edebilirsiniz")
    else:
        print("- Model performansını artırmak için mimariyi değiştirmeyi deneyin")
    
    return slope #hatalardaki artış eğimi







def getOutput():
    output_values = []
    max_value = -1
    max_index = -1
    
    # Tüm çıktı nöronlarını işle
    for i, neuron in enumerate(layers[-1]):
        value = neuron.value
        weighted_sum = neuron.weightedSum
        
        # En yüksek aktivasyonu takip et
        if value > max_value:
            max_value = value
            max_index = i
        
        output_values.append(f"Nöron {i}: Değer={value:.8f}, AğırlıklıToplam={weighted_sum:.4f}")
    
    # En yüksek aktivasyon bilgisini ekle
    if max_index != -1:
        output_values.append(f"\nSONUÇ: {max_index} (Olasılık: {max_value*100:.2f}%)")
    
    return output_values














def evaluate_network(X_test, y_test):
    """Modeli değerlendir"""
    correct = 0
    for i in range(len(X_test)):
        # Girişi ayarla
        for j in range(min(len(layers[0]), len(X_test[i]))):
            layers[0][j].value = X_test[i,j]
        
        runAI()
        
        # Tahmin
        outputs = [neuron.value for neuron in layers[-1][:len(y_test[i])]]
        predicted = np.argmax(outputs)
        actual = np.argmax(y_test[i])
        
        if predicted == actual:
            correct += 1
        
        if i % 50 == 0:
            print(f"Test {i}/{len(X_test)} - Doğruluk: {correct/(i+1):.2%}")
    
    print(f"\nFinal Test Doğruluk: {correct/len(X_test):.2%}")



bias_is=None
def disable_all_biases():
    global bias_is
    bias_is=False
    for layer_idx in connections:
        for neuron_id in connections[layer_idx]:
            for conn in connections[layer_idx][neuron_id]:
                conn.bias = 0

# Kullanım:
disable_all_biases()

def enable_all_biases():
    global bias_is
    bias_is=True
    for layer_idx in connections:
        for neuron_id in connections[layer_idx]:
            for conn in connections[layer_idx][neuron_id]:
                conn.bias = random.uniform(-0.1, 0.1)  # Rastgele küçük değerlerle yeniden başlat

# Kullanım:
#enable_all_biases()



"""
    [Başla]
       │
       ├─► (Opsiyonel) Başlangıç Checkpoint'u Oluştur  ✅
       │       └─ Modelin başlangıç durumunu kaydet (rollback için)
       │
       ├─► Backpropagation Çalıştır  ✅
       │
       ├─► Nöron Sağlığını Hesapla  ✅
       │       └─ Her nöronun aktivasyon, ağırlık dağılımı ve öğrenme potansiyeli gibi metriklerini değerlendir
       │            (Ayrıca nöron sağlığı puanlarını kaydet ve izleyerek hangi nöronların zayıf olduğunu belirle)
       │
       ├─► Kritik Nöronları Belirle ve Düzenle  
       │       └─ Zayıf nöronlar için:
       │             ├─ Gerekirse nöron ekle (daha fazla öğrenme kapasitesi için)
       │             ├─ Gerekirse nöron sil (fazla gürültü veya zayıf performans için)
       │             └─ Gerekirse nöronları birleştir veya böl
       │             └─ (Deneysel modda yapılan değişikliklerde geçici ekleme/silme uygulayarak performans karşılaştırması yap)
       │
       ├─► Katman Sağlığını Ölç  
       │       └─ Katman bazlı performans ve overfitting kontrolü yap
       │            (Her katmanın validasyon performansını da izleyerek genişleme/silme kararını destekle)
       │
       ├─► Katman Yapısını Dinamik Olarak Düzenle  
       │       └─ Gerekirse yeni katman ekle veya mevcut katmanları sil
       │            (Yapısal genişlemenin overfitting'e yol açıp açmadığını kontrol etmek için test/simülasyon uygulayın)
       │
       ├─► Değişiklikleri Kaydet  
       │       └─ Yapısal ve hiperparametre değişikliklerinin kaydını (checkpoint) oluştur  
       │            (Her önemli değişiklikten sonra modelin durumunu kaydedin)
       │
       ├─► Hiperparametreleri Dinamik Olarak Ayarla  
       │       └─ Özellikle learning rate ve regularization parametrelerini, loss trendine ve nöron/katman sağlık metriklerine göre artır/azalt
       │            (Entegre kontrol algoritmasıyla, örneğin “patience” ve “stability window” kullanarak karar verin)
       │
       ├─► Kademeli ve Ölçülebilir Karar Mekanizması Uygula  
       │       └─ Değişiklikleri her epoch’da değil, belirli bir süre boyunca (örneğin, birkaç epoch) izleyerek
       │             • Geçici dalgalanmaların etkisinden kaçınmak için iyileşme eşiği (threshold) belirle
       │             • Rollback zaman penceresini tanımla
       │
       ├─► Belirli Bir Süre (Stability Window) Bekle ve Gözlemle  ✅
       │       └─ Yapılan değişikliklerin etkisini birkaç epoch boyunca değerlendir  
       │             (Bu süre zarfında modelin performansı, nöron ve katman sağlığı metrikleri detaylıca izlenir)
       │
       ├─► Simülasyon ve A/B Testleri Uygula (Opsiyonel)  
       │       └─ Geliştirilen dinamik yapı düzenleme algoritmasını küçük ölçekli testlerle dene
       │             • Farklı stratejilerin performansını karşılaştırarak en etkili olanları seç
       │
       └─► Hata Arttı mı?  
               ├─► Evet → Geri Al (Rollback: Önceki Checkpoint'e dön ve değişiklikleri geri çek)
               └─► Hayır → [Devam]
    """

def is_multiple(a, b):
    return a % b == 0 if b != 0 else False

def detect_u_turn(loss_history, check_window=20, delta=1e-4):
    """
    Loss değerlerinin önce düştüğünü sonra yükseldiğini tespit eder.
    
    Args:
        loss_history (list of float): Kayıtlı loss değerleri.
        check_window (int): U-dönüşü kontrol etmek için geçmişteki kaç epoch’a bakılacak.
        delta (float): Azalma ve artma algılama hassasiyeti.

    Returns:
        bool: Eğer U-dönüşü varsa True, yoksa False.
        dict: Ek bilgiler (minimum nokta, hangi epoch’ta oldu, ortalama değişimler).
    """
    if len(loss_history) < 2 * check_window:
        return False, {}

    recent = loss_history[-check_window:]
    before = loss_history[-2*check_window:-check_window]

    avg_before = sum(before) / len(before)
    avg_recent = sum(recent) / len(recent)

    min_loss = min(loss_history[-2*check_window:])
    min_index = loss_history[-2*check_window:].index(min_loss)
    absolute_index = len(loss_history) - 2*check_window + min_index

    decreasing = avg_before - min_loss > delta
    increasing = avg_recent - min_loss > delta

    u_turn_detected = decreasing and increasing

    return u_turn_detected, {
        "u_turn_detected": u_turn_detected,
        "min_loss": min_loss,
        "min_loss_epoch": absolute_index,
        "avg_before": avg_before,
        "avg_recent": avg_recent,
        "decrease": avg_before - min_loss,
        "increase": avg_recent - min_loss
    }


class CorticalColumn:
    
    def __init__(self, log_file="network_changes.log", learning_rateArg=0.3,targetError=None,
                 maxEpochForTargetError=1000,originalNetworkModel=None,
                 overfit_threshold=0.9):
        global layers, connections
        
        self.firstLearningRate = learning_rateArg
        self.learningRate = learning_rateArg
        self.neuronHealtThreshould = 0.3
        self.change_log = []  # Değişiklik loglarını tutacak liste
        self.current_epoch = 0  # Mevcut epoch bilgisi
        self.log_file = log_file
        self.log_start_time = time.time()
        # Loss history'yi de burada tutmak isterseniz:
        self.loss_history = []

        self.lr_cooldown = 0
        self.lr_cooldown_period = 20  # Increased from 10
        self.last_lr_change_epoch = -float('inf')
        self.lrChanged = 0

        self.maxEpochForTargetError=maxEpochForTargetError
        self.targetError=targetError
        if debug and self.targetError != None:print(f"Hedef hata değerine erişilememesi durumunda minimum hata değerinin sıfırlanması için gereken epoch sayısı {self.maxEpochForTargetError} değerinin katları olarak belirlendi.")
        self.originalNetworkModel = originalNetworkModel
        self.neuron_health_history = {}  # Dictionary to store health history of neurons

                # Overfitting control
        self.overfit_threshold = overfit_threshold
        self.val_error_history = []

        
        

        # Log dosyasını başlat (varsa sil, yenisini oluştur)
        with open(self.log_file, 'w') as f:
            f.write("")  # Boş dosya oluştur

    def _get_timestamp(self):
        """Geçerli zaman damgasını ISO formatında döndürür"""
        return datetime.now().isoformat()
    
    def _append_to_log(self, entry):
        """Log girişini hem bellekte hem de dosyaya ekler"""
        self.change_log.append(entry)
        with open(self.log_file, 'a') as f:
            json.dump(entry, f)
            f.write("\n")

    def log_change(self, change_type, details):
        global layers
        """Değişiklikleri loglayan yardımcı fonksiyon"""
        log_entry = {
            'epoch': self.current_epoch,
            'timestamp': self._get_timestamp(),
            'elapsed_seconds': round(time.time() - self.log_start_time, 2),
            'type': change_type,
            'details': details,
            'network_state': {
                'layer_sizes': [len(layer) for layer in layers],
                'total_neurons': sum(len(layer) for layer in layers)
                
            }
        }
        self._append_to_log(log_entry)

    def calculate_slope(self, error_history, start_epoch=None, end_epoch=None):
        """
        Belirtilen epoch aralığındaki hata değerlerinin eğimini hesaplar

        Parametreler:
        error_history (list): Hata değerlerinin listesi
        start_epoch (int): Başlangıç epoch indeksi (None ise son %20'nin başlangıcı)
        end_epoch (int): Bitiş epoch indeksi (None ise son epoch)

        Returns:
        float: Hata eğimi (pozitif = hatalar artıyor, negatif = hatalar azalıyor)
        float: R² değeri (eğimin güvenilirliği, 1'e yakın = güvenilir)
        """
        if not error_history or len(error_history) < 2:
            return 0.0, 0.0

        # Varsayılan aralıkları ayarla
        if start_epoch is None:
            start_epoch = int(len(error_history) * 0.8)  # Son %20'nin başlangıcı
        if end_epoch is None:
            end_epoch = len(error_history) - 1  # Son epoch

        # Geçerli aralığı kontrol et
        start_epoch = max(0, min(start_epoch, len(error_history)-1))
        end_epoch = max(0, min(end_epoch, len(error_history)-1))

        if start_epoch >= end_epoch:
            return 0.0, 0.0

        # Seçilen aralıktaki hata ve epoch değerlerini al
        selected_errors = error_history[start_epoch:end_epoch+1]
        epochs = list(range(start_epoch, end_epoch+1))

        # Lineer regresyon ile eğim ve R² değerini hesapla
        slope, intercept, r_value, _, _ = stats.linregress(epochs, selected_errors)
        r_squared = r_value**2

        return slope, r_squared

    def monitor_network(self, avg_error):
        """
        Enhanced monitoring with plateau detection and improved learning rate adjustments.
        This function analyzes error trends to manage learning rate changes for optimal training.
        """
        # Add the current error to history
        self.loss_history.append(avg_error)
        window_size = min(70, len(self.loss_history))
        startEpoch=len(self.loss_history) - window_size
        slope, r_square = self.calculate_slope(
                self.loss_history, 
                start_epoch=startEpoch, 
                end_epoch=None
        )

        if slope<1e-6 and slope >-1e-6:
            self.log_change('its linear now',{
                'between':f"{startEpoch}-now",
                'slope':slope
            })
        


        #print(is_multiple(self.current_epoch,self.maxEpochForTargetError),slope)
        if is_multiple(self.current_epoch+1,self.maxEpochForTargetError) and slope > -1e-8:
            minError = np.min(self.loss_history)
            increaseValue=0.9
            if debug:
                self.log_change('minimum Error Changed and Neural network resetted',{
                    'current target error':self.targetError,
                    'minimum Error':minError,
                    'now error':self.loss_history[-1],
                    'current lr':self.learningRate,
                    'first lr':self.firstLearningRate,
                    'increase value':increaseValue,
                    'now target value':minError/increaseValue

                })
            self.learningRate = self.firstLearningRate
            self.targetError = minError/increaseValue
            #print("---------------------------------------------")
            if self.originalNetworkModel is not None:
                setLayers(self.originalNetworkModel)
            return self.learningRate,self.targetError
            

        # Log debug information about error trend if needed
        if debug:
            if len(self.loss_history) % 50 == 0 :
                
                self.log_change('error_trend_analysis', {
                    'window_size': window_size,
                    'slope': slope,
                    'r_square': r_square,
                    'current_lr': self.learningRate,
                })
        
        
        
        # Check if we can update learning rate (not in cooldown period)
        if self.current_epoch - self.last_lr_change_epoch >= self.lr_cooldown_period:
            # Apply a small automatic decay to prevent stagnation (0.9995^1000 ≈ 0.61, so gradual)
            # This helps ensure the learning rate doesn't stay too high for too long
            
            # Get new learning rate based on loss trend
            new_lr = self.update_learning_rate(self.learningRate, self.loss_history,slopeArg=slope)
            
            # If learning rate changed significantly, reset cooldown counter

            self.learningRate = new_lr
            self.last_lr_change_epoch = self.current_epoch


        if self.current_epoch % 10 == 0:  # Her 10 epoch'ta bir
            self.adapt_network_structure(avg_error)
        
        return self.learningRate,None
    
    def update_learning_rate(self, current_lr, loss_history, 
                             patience=60, # Increased from 20
                             min_lr=0.1, max_lr=4.0,  # Reduced max_lr
                             factor=0.005,  # Reduced from 0.002
                             threshold=1e-1, increase_threshold=5e-2,slopeArg=None):  # More conservative increase threshold
        """
        Updates learning rate based on loss history trends with improved stability.
        
        Parameters:
        - current_lr: Current learning rate
        - loss_history: History of loss values
        - patience: Number of epochs to consider for trend analysis
        - min_lr/max_lr: Bounds for learning rate
        - factor: Rate of change factor for learning rate adjustments
        - threshold: Minimum improvement threshold to maintain current lr
        - increase_threshold: Threshold to consider increasing lr
        
        Returns:
        - Updated learning rate value
        """




        # Return current_lr if we don't have enough data
        if len(loss_history) < 2 * patience:
            return current_lr
        
            # Yeni: Gradyan Stabilizasyonu
        if abs(slopeArg) > 0.1:  # Ani hata değişimlerinde
            emergency_factor = 0.2 if slopeArg > 0 else 0.1
            new_lr = current_lr * (1 - emergency_factor)
            self.log_change('emergency change bc slope',{
                'slope':slopeArg,
                'emergency_factor':emergency_factor,
                'old_lr':current_lr,
                'new_lr':new_lr,
                'change':new_lr-current_lr

            })
            return max(new_lr, min_lr)
        
        # Analyze recent and previous loss trends
        recent_losses = loss_history[-patience:]
        previous_losses = loss_history[-(2 * patience):-patience]
        
        # Yeni: Cosine Annealing Esintili Decay
        cosine_decay = 0.5 * (1 + math.cos(math.pi * self.current_epoch / 1000))
        decayed_lr = current_lr * cosine_decay
        
        # Calculate average losses
        avg_old = sum(previous_losses) / patience if previous_losses else 0
        avg_new = sum(recent_losses) / patience if recent_losses else 0
        
        # Calculate relative improvement
        improvement = (avg_old - avg_new) / avg_old if avg_old != 0 else 0
        
        # Check for error increase (negative improvement)
        if improvement < -0.01:  # More sensitive to error increases (1% worsening)
            # Error is increasing - reduce learning rate more aggressively
            reduction_factor = factor * 3  # More aggressive reduction than before
            new_lr = current_lr * (1 - reduction_factor)
            new_lr = max(new_lr, min_lr)
            
            self.log_change('lr_emergency_reduction', {
                'before_lr': current_lr,
                'new_lr': new_lr,
                'change': new_lr - current_lr,
                'reason': f"Error increasing: improvement={improvement}",
                'lrChanged': self.lrChanged
            })
            
            # Reset change counter when we make an emergency reduction
            self.lrChanged = -1
            
            print(f"Error increasing! Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
            return new_lr
        
        # Case 1: Insufficient improvement - reduce learning rate
        if improvement < threshold:
            # Check if we're at minimum learning rate already
            if current_lr <= min_lr * 1.01:
                # At minimum, try a small increase to escape potential local minimum
                # Less aggressive than before
                small_increase_factor = 1.5  # Was 1/(factor*10) which was too large
                new_lr = current_lr * small_increase_factor
                new_lr = min(new_lr, max_lr)
                
                self.log_change('lr_small_up', {
                    'before_lr': current_lr,
                    'new_lr': new_lr,
                    'change': new_lr - current_lr,
                    'increase_factor': small_increase_factor,
                    'reason': f'min_limit_reached, trying escape',
                    'lrChanged': self.lrChanged
                })
                
                self.lrChanged = 0  # Reset change counter
                print(f"Min limit reached. Learning rate slightly increased: {current_lr:.6f} -> {new_lr:.6f}")
                return new_lr
            else:
                # Standard reduction
                new_lr = current_lr * (1 - factor)
                new_lr = max(new_lr, min_lr)
                
                self.log_change('lr_down', {
                    'before_lr': current_lr,
                    'new_lr': new_lr,
                    'change': new_lr - current_lr,
                    'factor': factor,
                    'reason': f"improvement: {improvement:.6f} < threshold: {threshold}",
                    'lrChanged': self.lrChanged
                })
                
                # Track consecutive decreases
                if self.lrChanged <= 0:
                    self.lrChanged -= 1
                else:
                    self.lrChanged = -1
                    
                print(f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f} (improvement: {improvement:.6f})")
                return new_lr
        
        # Case 2: Strong improvement - consider increasing learning rate
        elif improvement > increase_threshold:
            # Check if we're at maximum learning rate already
            if current_lr >= max_lr * 0.999:
                # At maximum, try a small decrease to prevent divergence
                reduction_factor = factor * 10  # Less aggressive than before (was 50)
                new_lr = current_lr * (1 - reduction_factor)
                new_lr = max(new_lr, min_lr)
                
                self.log_change('lr_max_down', {
                    'before_lr': current_lr,
                    'new_lr': new_lr,
                    'change': new_lr - current_lr,
                    'reduction_factor': reduction_factor,
                    'reason': f'max_limit_reached, preventing divergence',
                    'lrChanged': self.lrChanged
                })
                
                self.lrChanged = 0  # Reset change counter
                print(f"Max limit reached. Learning rate decreased: {current_lr:.6f} -> {new_lr:.6f}")
                return new_lr
            else:
                # Standard increase, but more conservative now
                # Adaptive factor - increase more conservatively if we're already at a high learning rate
                adaptive_factor = factor * 0.5 * (1 - (current_lr / max_lr) * 0.8)  # More conservative increase
                new_lr = current_lr * (1 + adaptive_factor)
                new_lr = min(new_lr, max_lr)
                
                self.log_change('lr_up', {
                    'before_lr': current_lr,
                    'new_lr': new_lr,
                    'change': new_lr - current_lr,
                    'adaptive_factor': adaptive_factor,
                    'reason': f"strong improvement: {improvement:.6f} > threshold: {increase_threshold}",
                    'lrChanged': self.lrChanged
                })
                
                # Track consecutive increases
                if self.lrChanged >= 0:
                    self.lrChanged += 1
                else:
                    self.lrChanged = 1
                    
                print(f"Learning rate increased: {current_lr:.6f} -> {new_lr:.6f} (improvement: {improvement:.6f})")
                return new_lr
        
        # Case 3: Moderate improvement - maintain current learning rate with tiny decay
        if debug:
            print(f"Learning rate only slightly decayed: {current_lr:.6f} -> {decayed_lr:.6f} (improvement: {improvement:.6f})")
        # Mevcut mantığı cosine decay ile birleştir
        return max(decayed_lr * 0.99, min_lr)  # Ekstra yavaş decay

    def backpropagation(self,input_data, target_data):
        global layers, connections

        # 1. İleri Besleme - Giriş verilerini ağa ver
        # Giriş katmanındaki nöron değerlerini ayarla
        for i, value in enumerate(input_data):
            if i < len(layers[0]):
                layers[0][i].value = value

        # İleri besleme işlemi - tüm ağı hesapla

        runAI()

        # 2. Hata Hesaplama
        # Çıkış katmanındaki her nöron için hata hesapla
        output_layer = layers[-1]
        output_errors = []

        for i, neuron in enumerate(output_layer):
            if i < len(target_data):
                error = target_data[i] - neuron.value
                output_errors.append(error)
            else:
                output_errors.append(0)  # Hedef veri yoksa hata 0

        # 3. Geri Yayılım
        # Her katman için delta değerlerini hesapla (çıkıştan girişe doğru)
        deltas = [[] for _ in range(len(layers))]

        # Önce çıkış katmanındaki delta değerlerini hesapla
        for i, neuron in enumerate(output_layer):
            if i < len(output_errors):
                # Delta = Hata * Aktivasyon fonksiyonunun türevi
                delta = output_errors[i] * neuron.activation_derivative()
                deltas[-1].append(delta)
            else:
                deltas[-1].append(0)

        # Gizli katmanlar için delta değerlerini hesapla (geriye doğru)
        for layer_idx in range(len(layers)-2, 0, -1):  # Son gizli katmandan ilk gizli katmana
            for i, neuron in enumerate(layers[layer_idx]):
                error = 0
                # Bu nörondan sonraki katmana olan tüm bağlantıları kontrol et
                for conn in connections[layer_idx].get(neuron.id, []):
                    # Sonraki katmandaki nöronu bul
                    next_layer_idx = layer_idx + 1
                    for j, next_neuron in enumerate(layers[next_layer_idx]):
                        if conn.connectedTo[1] == next_neuron.id:
                            # Bu bağlantının ağırlığı * sonraki nöronun deltası
                            error += conn.weight * deltas[next_layer_idx][j]
                            break
                        
                # Delta = Hata * Aktivasyon fonksiyonunun türevi
                delta = error * neuron.activation_derivative()
                deltas[layer_idx].append(delta)

        # 4. Ağırlık Güncelleme
        for layer_idx in range(len(layers)-1):
            for i, neuron in enumerate(layers[layer_idx]):
                for conn in connections[layer_idx].get(neuron.id, []):
                    # Bir sonraki katmandaki bağlı nöronu bul
                    next_layer_idx = layer_idx + 1
                    for j, next_neuron in enumerate(layers[next_layer_idx]):
                        if conn.connectedTo[1] == next_neuron.id:
                            # Ağırlık değişimi = öğrenme oranı * delta * nöron çıktısı
                            
                            if debug:
                                weight_change = self.learningRate * deltas[next_layer_idx][j] * neuron.value
                                old_weight = conn.weight
                                #print("debug values :",weight_change,self.learningRate,deltas[next_layer_idx][j],neuron.value)
                            # Mevcut Connection sınıfınızdaki update_weight metodunu kullan
                            conn.update_weight(self.learningRate, deltas[next_layer_idx][j] * neuron.value)
                            break
                        
        # Toplam hata değerini hesapla ve döndür (MSE - Mean Squared Error)
        total_error = sum(error**2 for error in output_errors) / len(output_errors) if output_errors else 0
        


        if debug:
            # Ağırlık güncellemelerini logla
            weight_updates = []
            for layer_idx in range(len(layers)-1):
                for i, neuron in enumerate(layers[layer_idx]):
                    for conn in connections[layer_idx].get(neuron.id, []):
                        weight_updates.append({
                            'from_neuron': conn.connectedTo[0],
                            'to_neuron': conn.connectedTo[1],
                            'old_weight': old_weight,  # Güncellemeden önceki ağırlık
                            'new_weight': conn.weight,
                            'change': conn.weight-old_weight
                        })

            if weight_updates:
                self.log_change('weight_updates', {
                    'count': len(weight_updates),
                    'average_change': sum(abs(w['change']) for w in weight_updates) / len(weight_updates),
                    'updates': weight_updates[:10]  # İlk 10 güncellemeyi göster (performans için)
                })

        return total_error
        
    def adapt_neurons(self):
        pass
                
    
    def calculate_neuron_health(self,neuron):
        global layers,connections
        # 1. Nöronun aktivasyon değeri
        activation_score = neuron.output

        # 2. Bağlantıların ağırlık değerlerini kontrol et
        weight_sum = 0
        weight_count = 0

        # Gelen bağlantıları bul
        for layer_idx in range(len(layers) - 1):
            for prev_neuron in layers[layer_idx]:
                for conn in connections[layer_idx].get(prev_neuron.id, []):
                    if conn.connectedTo[1] == neuron.id:
                        weight_sum += abs(conn.weight)  # Mutlak değer kullan
                        weight_count += 1

        # Giden bağlantıları bul (eğer bu nöron çıkış katmanında değilse)
        current_layer_idx = None
        for layer_idx, layer in enumerate(layers):
            if any(n.id == neuron.id for n in layer):
                current_layer_idx = layer_idx
                break
            
        if current_layer_idx is not None and current_layer_idx < len(layers) - 1:
            for conn in connections.get(current_layer_idx, {}).get(neuron.id, []):
                weight_sum += abs(conn.weight)
                weight_count += 1

        # Ortalama ağırlık (eğer bağlantı varsa)
        avg_weight = weight_sum / weight_count if weight_count > 0 else 0

        # 3. Aktivasyon türevi - nöronun ne kadar öğrenmeye açık olduğunu gösterir
        learning_potential = neuron.activation_derivative()

        # Tüm faktörleri birleştirerek bir sağlık puanı hesapla
        # Bu formülü kendi ihtiyaçlarınıza göre ayarlayabilirsiniz
        health_score = (
            0.4 * activation_score +  # Aktivasyon değerine %40 ağırlık ver
            0.4 * avg_weight +        # Ortalama ağırlığa %40 ağırlık ver
            0.2 * learning_potential  # Öğrenme potansiyeline %20 ağırlık ver
        )

        return health_score
    

        


    def find_neuron_layer(self,neuron_id):
        """Verilen ID'ye sahip nöronun hangi katmanda olduğunu bulur"""
        global layers
        for layer_idx, layer in enumerate(layers):
            for neuron in layer:
                if neuron.id == neuron_id:
                    return layer_idx,neuron
        return None

    def add_neuron_to_layer(self,layer_index=None, neuron_id=None):
        """
        Belirli bir katmana yeni bir nöron ekler ve sadece bu nöronla ilgili bağlantıları oluşturur

        Args:
            layer_index: Nöronun ekleneceği katmanın indeksi (None ise neuron_id'nin katmanı kullanılır)
            neuron_id: Referans nöron ID'si (bunun katmanına yeni nöron eklenir, layer_index None ise)
            activation_type: Yeni nöronun aktivasyon fonksiyonu tipi
        """
        global layers, connections

        # Eğer layer_index verilmediyse ve neuron_id verildiyse, nöronun katmanını bul
        if layer_index is None and neuron_id is not None:
            layer_index = self.find_neuron_layer(neuron_id)
            if layer_index is None:
                print(f"Hata: ID'si {neuron_id} olan nöron bulunamadı.")
                return None

        # Hala layer_index belirlenemedi ise, varsayılan olarak son gizli katmanı kullan
        if layer_index is None:
            if len(layers) <= 2:  # Sadece giriş ve çıkış katmanları varsa
                layer_index = 0  # Giriş katmanına ekle (veya tercih ettiğiniz başka bir strateji)
            else:
                layer_index = len(layers) - 2  # Son gizli katman (çıkış katmanından bir önceki)

            if debug:
                print(f"Katman indeksi belirlenmediği için varsayılan olarak katman {layer_index} kullanılıyor.")

        if layer_index < 0 or layer_index >= len(layers):
            print(f"Hata: {layer_index} indeksi geçerli bir katman indeksi değil.")
            return None

        # Yeni nöron oluştur
        new_neuron = Neuron(activation_type=defaultNeuronActivationType)

        # Yeni nöronu katmana ekle
        layers[layer_index].append(new_neuron)

        # Önceki katman varsa, önceki katmandan bu nörona bağlantılar oluştur
        if layer_index > 0:
            prev_layer_idx = layer_index - 1
            for prev_neuron in layers[prev_layer_idx]:
                weight =random.uniform(-1/np.sqrt(len(layers[layer_index])), 
                                    1/np.sqrt(len(layers[layer_index])))
                conn = Connection(connectedToArg=[prev_neuron.id, new_neuron.id], weight=weight)

                if prev_neuron.id not in connections[prev_layer_idx]:
                    connections[prev_layer_idx][prev_neuron.id] = []

                connections[prev_layer_idx][prev_neuron.id].append(conn)

        # Sonraki katman varsa, bu nörondan sonraki katmana bağlantılar oluştur
        if layer_index < len(layers) - 1:
            for next_neuron in layers[layer_index + 1]:
                weight = random.uniform(-1/np.sqrt(len(layers[layer_index])), 
                                    1/np.sqrt(len(layers[layer_index])))
                conn = Connection(connectedToArg=[new_neuron.id, next_neuron.id], weight=weight)

                if new_neuron.id not in connections[layer_index]:
                    connections[layer_index][new_neuron.id] = []

                connections[layer_index][new_neuron.id].append(conn)

        if debug:
            print(f"Katman {layer_index}'e yeni nöron (ID: {new_neuron.id}) eklendi.")

        return new_neuron

    def remove_neuron_from_layer(self,layer_index=None, neuron_id=None):
        """
        Belirli bir nöronu ve sadece onunla ilgili bağlantıları siler

        Args:
            layer_index: Nöronun bulunduğu katmanın indeksi (None ise otomatik bulunur)
            neuron_id: Silinecek nöronun ID'si
        """
        global layers, connections

        if neuron_id is None:
            print("Hata: Silinecek nöronun ID'si belirtilmedi.")
            return False

        # Eğer layer_index verilmediyse, nöronun katmanını bul
        if layer_index is None:
            layer_index = self.find_neuron_layer(neuron_id)
            if layer_index is None:
                print(f"Hata: ID'si {neuron_id} olan nöron bulunamadı.")
                return False

        if layer_index < 0 or layer_index >= len(layers):
            print(f"Hata: {layer_index} indeksi geçerli bir katman indeksi değil.")
            return False

        # Nöronu bul
        neuron_to_remove = None
        for i, neuron in enumerate(layers[layer_index]):
            if neuron.id == neuron_id:
                neuron_to_remove = neuron
                neuron_index = i
                break
            
        if neuron_to_remove is None:
            print(f"Hata: ID'si {neuron_id} olan nöron, katman {layer_index}'de bulunamadı.")
            return False

        # Nöronu katmandan çıkar
        layers[layer_index].pop(neuron_index)

        # Önceki katmandan bu nörona gelen bağlantıları sil
        if layer_index > 0:
            prev_layer_idx = layer_index - 1
            for prev_neuron_id in list(connections[prev_layer_idx].keys()):
                connections[prev_layer_idx][prev_neuron_id] = [
                    conn for conn in connections[prev_layer_idx][prev_neuron_id] 
                    if conn.connectedTo[1] != neuron_id
                ]

        # Bu nörondan sonraki katmana giden bağlantıları sil
        if layer_index < len(layers) - 1:
            if neuron_id in connections[layer_index]:
                del connections[layer_index][neuron_id]

        if debug:
            print(f"Katman {layer_index}'den nöron (ID: {neuron_id}) silindi.")

        return True


    
    def adapt_network_structure(self, avg_train_error, avg_val_error=None):
        """
        Dinamik yapı adaptasyonu:
        - Overfitting kontrolü ve gerekirse nöron kırpma
        - Nöron seviyesinde optimizasyon
        - Katman seviyesinde optimizasyon
        - Stratejik büyüme
        """
        # 0. Overfitting kontrolü
        if avg_val_error is not None:
            self.val_error_history.append(avg_val_error)
            gap = avg_val_error - avg_train_error
            if gap > self.overfit_threshold:
                self.prune_for_overfitting(gap)
                return

        # 1. Nöron seviyesinde optimizasyon
        self.neuron_level_optimization(avg_train_error)
        # 2. Katman seviyesinde optimizasyon
        self.layer_level_optimization(avg_train_error)
        # 3. Stratejik büyüme
        self.strategic_growth(avg_train_error)

    def prune_for_overfitting(self, gap):
        """
        Aşırı öğrenme tespit edilirse, nöron sayısını azaltarak basitleştirir
        """
        global layers
        # Oran: gap / overfit_threshold, en fazla %50 azaltma
        factor = min(gap / self.overfit_threshold, 1.0) * 0.5
        for idx in range(1, len(layers) - 1):
            layer = layers[idx]
            current_size = len(layer)
            desired_size = max(2, int(current_size * (1 - factor)))
            to_remove = current_size - desired_size
            if to_remove > 0:
                # Sağlık skoru düşük nöronları öncelikli kaldır
                scores = [(neuron, self.calculate_neuron_health(neuron)) for neuron in layer]
                scores.sort(key=lambda x: x[1])
                for neuron, _ in scores[:to_remove]:
                    self.remove_neuron_from_layer(idx, neuron.id)
                    self.log_change('pruned_for_overfit', {
                        'neuron_id': neuron.id,
                        'layer': idx,
                        'gap': gap,
                        'reason': 'Overfitting pruning'
                    })
    
    def neuron_level_optimization(self, avg_error):
        """
        Her katmandaki nöronları değerlendirir ve gereksiz olanları kaldırır
        """
        global layers
        
        # Sağlık eşik değerini hataya göre dinamik olarak ayarla
        health_threshold = max(0.2, min(0.5, 0.3 * (1 + avg_error)))
        
        for layer_idx, layer in enumerate(layers):
            # Katmanın çıktı katmanı olup olmadığını kontrol et
            is_output_layer = (layer_idx == len(layers) - 1)
            
            for neuron in layer[:]:  # Kopya üzerinde döngü
                health = self.calculate_neuron_health(neuron)
                
                # Sağlık skoru eşik değerin altındaysa ve çıktı katmanında değilse
                if health < health_threshold and not is_output_layer:
                    # Nöronun son N sağlık değerini kontrol et
                    if len(self.neuron_health_history.get(neuron.id, [])) > 3:
                        last_3_health = self.neuron_health_history[neuron.id][-3:]
                        if all(h < health_threshold for h in last_3_health):
                            self.remove_neuron_from_layer(layer_idx, neuron.id)
                            self.log_change('neuron_removed', {
                                'neuron_id': neuron.id,
                                'layer': layer_idx,
                                'health': health,
                                'reason': f'Low health (<{health_threshold:.2f})'
                            })
    
    def layer_level_optimization(self, avg_error):
        """
        Katmanları değerlendirir ve gereksiz olanları kaldırır
        """
        global layers
        
        if len(layers) <= 2:  # En az giriş ve çıkış katmanı olmalı
            return
            
        # Gizli katmanların sağlığını hesapla
        layer_health_scores = []
        for layer_idx in range(1, len(layers)-1):  # Gizli katmanlar
            layer_health = self.calculate_layer_health(layer_idx)
            layer_health_scores.append((layer_idx, layer_health))
            
        # En düşük sağlıklı katmanı bul
        layer_health_scores.sort(key=lambda x: x[1])
        worst_layer_idx, worst_health = layer_health_scores[0]
        
        # Katman sağlık eşiğini hataya göre dinamik ayarla
        layer_health_threshold = max(0.3, min(0.6, 0.4 * (1 + avg_error)))
        
        # Eşik değerin altındaysa ve yeterli katman varsa sil
        if worst_health < layer_health_threshold and len(layers) > 3:
            self.remove_layer(worst_layer_idx)
            self.log_change('layer_removed', {
                'layer_idx': worst_layer_idx,
                'health': worst_health,
                'reason': f'Low layer health (<{layer_health_threshold:.2f})'
            })
    
    def strategic_growth(self, avg_error):
        """
        Ağın büyümesini stratejik olarak yönetir:
        - Zayıf katmanları güçlendirir
        - Kritik bölgelere nöron ekler
        - Gerektiğinde yeni katman ekler
        """
        # 1. Zayıf katmanları güçlendir
        self.strengthen_weak_layers(avg_error)
        
        # 2. Kritik bölgelere nöron ekle
        self.add_neurons_to_critical_areas()
        
        # 3. Gerekirse yeni katman ekle
        self.add_layer_if_needed(avg_error)
    
    def strengthen_weak_layers(self, avg_error):
        """
        Zayıf katmanlara nöron ekler
        """
        global layers
        
        complexity_factor = max(0.4, min(0.8, 0.5 * (1 + avg_error)))
        
        for layer_idx in range(1, len(layers)-1):  # Gizli katmanlar
            layer_health = self.calculate_layer_health(layer_idx)
            if layer_health < complexity_factor:
                # Katmana 1-2 nöron ekle
                num_neurons_to_add = 1 if len(layers[layer_idx]) < 10 else 2
                for _ in range(num_neurons_to_add):
                    new_neuron = self.add_neuron_to_layer(layer_index=layer_idx)
                    self.log_change('neuron_added', {
                        'neuron_id': new_neuron.id,
                        'layer': layer_idx,
                        'reason': f'Strengthening weak layer (health={layer_health:.2f})'
                    })
    
    def add_neurons_to_critical_areas(self):
        """
        Yüksek hata üreten veya yüksek öğrenme potansiyeli olan bölgelere nöron ekler
        """
        global layers
        
        # En yüksek aktivasyon türevine sahip nöronun katmanına ekleme yap
        max_derivative = -1
        target_layer = None
        
        for layer_idx, layer in enumerate(layers):
            for neuron in layer:
                derivative = neuron.activation_derivative()
                if derivative > max_derivative:
                    max_derivative = derivative
                    target_layer = layer_idx
                    
        if target_layer is not None and target_layer < len(layers)-1 and target_layer != 0:
            new_neuron = self.add_neuron_to_layer(layer_index=target_layer)
            self.log_change('neuron_added', {
                'neuron_id': new_neuron.id,
                'layer': target_layer,
                'reason': f'High learning potential (derivative={max_derivative:.2f})'
            })
    
    def add_layer_if_needed(self, avg_error):
        """
        Ağın karmaşıklığı yeterli değilse yeni katman ekler
        """
        global layers
        
        if len(layers) >= 5:  # Maksimum 5 katman (giriş + 3 gizli + çıkış)
            return
            
        # Ortalama katman sağlığını hesapla
        total_health = 0
        for layer_idx in range(1, len(layers)-1):
            total_health += self.calculate_layer_health(layer_idx)
        avg_health = total_health / (len(layers)-2) if len(layers) > 2 else 0
        
        complexity_factor = max(0.4, min(0.7, 0.5 * (1 + avg_error)))
        
        if avg_health < complexity_factor * 0.7:  # Katmanlar çok yüklüyse
            # En yüklü katmanı bul
            max_load = -1
            busiest_layer = None
            for layer_idx in range(1, len(layers)-1):
                load = len(layers[layer_idx]) * self.calculate_connection_density(layer_idx)
                if load > max_load:
                    max_load = load
                    busiest_layer = layer_idx
                    
            if busiest_layer is not None:
                new_layer_idx = self.insert_hidden_layer(busiest_layer + 1)  # Meşgul katmandan sonra ekle
                self.log_change('layer_added', {
                    'layer_idx': new_layer_idx,
                    'reason': f'High layer load (load={max_load:.2f}, avg_health={avg_health:.2f})'
                })
    
    def calculate_layer_health(self, layer_idx):
        """
        Bir katmanın genel sağlık skorunu hesaplar
        """
        global layers
        
        layer = layers[layer_idx]
        if not layer:
            return 0
            
        # Katmandaki nöronların ortalama sağlığı
        total_health = sum(self.calculate_neuron_health(neuron) for neuron in layer)
        avg_neuron_health = total_health / len(layer)
        
        # Katmanın bağlantı yoğunluğu
        connection_density = self.calculate_connection_density(layer_idx)
        
        # Katmanın öğrenme potansiyeli (aktivasyon türevlerinin ortalaması)
        learning_potential = np.mean([neuron.activation_derivative() for neuron in layer])
        
        # Katman sağlık skoru
        layer_health = 0.5 * avg_neuron_health + 0.3 * connection_density + 0.2 * learning_potential
        
        return layer_health
    
    def calculate_connection_density(self, layer_idx):
        """
        Katmandaki bağlantı yoğunluğunu hesaplar
        """
        global layers, connections
        
        if layer_idx == 0:  # Giriş katmanı
            prev_layer_size = len(layers[layer_idx])
            current_layer_size = len(layers[layer_idx+1])
            total_possible = prev_layer_size * current_layer_size
        elif layer_idx == len(layers)-1:  # Çıkış katmanı
            return 1.0  # Çıkış katmanı için maksimum yoğunluk
        else:
            prev_layer_size = len(layers[layer_idx-1])
            current_layer_size = len(layers[layer_idx])
            next_layer_size = len(layers[layer_idx+1])
            total_possible = (prev_layer_size * current_layer_size) + (current_layer_size * next_layer_size)
        
        # Gerçek bağlantı sayısını hesapla
        actual_connections = 0
        if layer_idx > 0:
            for conn_list in connections[layer_idx-1].values():
                actual_connections += len(conn_list)
        
        if layer_idx < len(layers)-1:
            for conn_list in connections[layer_idx].values():
                actual_connections += len(conn_list)
                
        return actual_connections / total_possible if total_possible > 0 else 0
    
    def remove_layer(self, layer_idx):
        """
        Bir katmanı ve ilişkili bağlantıları kaldırır.
        Silme sonrası bağlantı sözlüğünü tutarlı hale getirmek için
        setConnections ile yeniden örülüyoruz.
        """
        global layers, connections

        if layer_idx <= 0 or layer_idx >= len(layers)-1:
            print("Hata: Giriş veya çıkış katmanı silinemez")
            return False

        # Katmanı kaldır
        del layers[layer_idx]

        # Eski connections anahtarlarını da temizle
        # (KeyError vermemesi için get ile guardlıyoruz)
        connections.pop(layer_idx-1, None)
        connections.pop(layer_idx,   None)

        # Kalan ağırlıkları koruyarak tüm bağlantıları yeniden inşa et
        # Böylece indekste kayma ya da eksik anahtar kalma riski ortadan kalkar
        setConnections(preserve_weights=True)

        if debug:
            print(f"Katman {layer_idx} silindi ve bağlantılar yeniden oluşturuldu.")
            

        return True

    
    def insert_hidden_layer(self, position):
        """
        Belirtilen pozisyona yeni bir gizli katman ekler
        """
        global layers, connections
        
        if position <= 0 or position >= len(layers):
            print("Hata: Geçersiz katman pozisyonu")
            return False
            
        # Yeni katman oluştur (mevcut katmanların ortalaması kadar nöronla)
        size = (len(layers[position-1]) + len(layers[position])) // 2
        new_layer = [Neuron(activation_type=defaultNeuronActivationType) for _ in range(max(2, size))]
        
        # Katmanı ekle
        layers.insert(position, new_layer)
        
        # Bağlantıları güncelle
        setConnections(True)
        
        # Önceki katmandan yeni katmana bağlantılar oluştur
        for prev_neuron in layers[position-1]:
            for new_neuron in new_layer:
                weight = np.random.uniform(-1, 1) * np.sqrt(2.0 / (len(layers[position-1]) + len(new_layer)))
                conn = Connection(connectedToArg=[prev_neuron.id, new_neuron.id], weight=weight)
                connections[position-1][prev_neuron.id].append(conn)
        
        # Yeni katmandan sonraki katmana bağlantılar oluştur
        for new_neuron in new_layer:
            for next_neuron in layers[position+1]:
                weight = np.random.uniform(-1, 1) * np.sqrt(2.0 / (len(new_layer) + len(layers[position+1])))
                conn = Connection(connectedToArg=[new_neuron.id, next_neuron.id], weight=weight)
                connections[position][new_neuron.id].append(conn)
        
        return position



# Terminal giriş döngüsü - Dinamik versiyon
while True:
    if cmd == "exit":
        break
        
    if cmd == "refresh()":
        clearGUI()
        runAI()
        visualize_network(layers, connections, refresh=True)
        print(getOutput())
    elif cmd=="history()":
        if debug:
            print(cmdHistory)
    elif cmd == "print_network()":
        # Ağ yapısını terminale yazdır
        print("\n=== AĞ YAPISI ===")
        for i, layer in enumerate(layers):
            print(f"\nKatman {i} ({len(layer)} nöron):")
            for neuron in layer:
                print(f"  Nöron ID: {neuron.id} | Değer: {neuron.value:.4f} | Aktivasyon: {neuron.activation_type}")
        
        # Bağlantıları yazdır
        print("\n=== BAĞLANTILAR ===")
        for layer_idx in connections:
            print(f"\nKatman {layer_idx} -> Katman {layer_idx+1}:")
            for src_id, conn_list in connections[layer_idx].items():
                for conn in conn_list:
                    print(f"  {src_id} → {conn.connectedTo[1]} | Ağırlık: {conn.weight:.4f}")
    
    elif cmd.startswith("get_connection("):
        try:
            args = cmd[15:-1].split(",")
            from_id = int(args[0])
            to_id = int(args[1])
            
            found = False
            for layer_idx in connections:
                if from_id in connections[layer_idx]:
                    for conn in connections[layer_idx][from_id]:
                        if conn.connectedTo[1] == to_id:
                            print(f"Bağlantı bilgisi: {from_id} → {to_id} | Ağırlık: {conn.weight:.6f}")
                            found = True
                            break
                    if found:
                        break
            if not found:
                print(f"Bağlantı bulunamadı: {from_id} → {to_id}")
                
        except Exception as error:
            print("Hatalı giriş! Örnek format: get_connection(5,10)")
            traceback.print_exc()

    

    elif cmd == "visualize()":
        visualizeNetwork = not visualizeNetwork
        print(f"Görselleştirme {'aktif' if visualizeNetwork else 'pasif'}")

    elif cmd.startswith("bias("):
        param=cmd[len("bias("):-1]
        if param == "True":
            enable_all_biases()
        elif param == "False":
            disable_all_biases()
        print("Bias is :",bias_is)


    elif cmd.startswith("loadModel("):
        print("Model dosyadan yükleniyor...")
        param=cmd[len("loadModel("):-1]
        try:
            load_network_optimized(param)
            print("Model başarıyla yüklendi.")
        except:
            print("Model yüklenirken hata oluştu. Model bozulmuş olabilir yeniden eğitim önerilir.")
            traceback.print_exc()
        
        pass
    elif cmd.startswith("train_custom("):
        stopEpoch = False
        try:
            params = cmd[len("train_custom("):-1].split(";")

            if not params[0].strip():
                raise ValueError("Dosya yolu gereklidir")

        # SADECE GEREKLİ PARAMETRELERİ ÇEK
            file_path = params[0].strip()
            network_structure = eval(params[1]) if len(params) > 1 and params[1].strip() else [2,4,1]
            print(list(network_structure))
            # DİĞER PARAMETRELERİ DİREKT FONKSİYONA PASLA (varsa)
            train_kwargs = {}
            if len(params) > 2 and params[2].strip():
                val = params[2].strip()
                train_kwargs['epochs'] = float(val) if '.' in val else int(val)
            if len(params) > 3 and params[3].strip():
                train_kwargs['batch_size'] = int(params[3])
            if len(params) > 4 and params[4].strip():
                train_kwargs['learning_rate'] = float(params[4])
            if len(params) > 5 and params[5].strip():
                train_kwargs['intelligenceValue'] = float(params[5])

            setLayers(list(network_structure))
            print(f"\nÖzel Eğitim Parametreleri:")
            print(f"- Dosya: {file_path}")
            print(f"- Ağ Yapısı: {network_structure}")
            for k, v in train_kwargs.items():
                print(f"- {k}: {v}")

            # Verileri yükle ve eğit
            X, y = modeltrainingprogram.read_csv_file(file_path)
            train_network(X, y, **train_kwargs)

        except Exception as e:
            print("Hatalı komut formatı! Örnek kullanımlar:")
            print("train_custom('veri.csv')  # Varsayılan parametrelerle (2-4-1 ağ)")
            print("train_custom('veri.csv';2,4,1)  # Ağ yapısı belirterek")
            print("train_custom('veri.csv';2,4,1;100)  # Epoch belirterek")
            print("train_custom('veri.csv';2,4,1;0.2)  # Hata değerini belirterek")
            print("train_custom('veri.csv';2,4,1;50;64;0.01)  # Tüm parametreler")
            print("train_custom('veri.csv';2,4,1;50;64;0.01;0.005)  # Intelligence ile")
            print("Parametre sırası: file_path;[network_structure];epochs;batch_size;learning_rate;intelligence")
            traceback.print_exc()

    elif (cmd.startswith("changeW(") and cmd.endswith(")")): 
        try:
            args = cmd[8:-1].split(";")
            from_id, to_id, new_weight = int(args[0]), int(args[1]), float(args[2])
            change_weight(connections, from_id, to_id, new_weight)
            print(f"Bağlantı ağırlığı güncellendi: {from_id} → {to_id} = {new_weight:.4f}")

        except Exception as error:
            print("Hatalı giriş! Örnek format: changeW(0;5;0.5)")
            traceback.print_exc()
            
    elif (cmd.startswith("changeN(") and cmd.endswith(")")): 
        try:
            args = cmd[8:-1].split(";")
            id, newValue = int(args[0]),float(args[1])
            get_neuron_by_id(id).value=newValue
            print(f"Nöron {id} değeri güncellendi: {newValue:.4f}")

        except Exception as error:
            print("Hatalı giriş! Örnek format: changeN(0;0.1)")
            traceback.print_exc()
    

    elif cmd.startswith("set_input"):
        try:
            # Komuttan değerleri ayır
            parts = cmd.split()

            if len(parts) < 2:
                print("Hata: Değerler gerekiyor. Örnek: set_input 0.1 0.5 0.9")
                print("Veya: set_input random (rastgele değerler için)")
                print("Veya: set_input zeros (sıfırlarla doldurmak için)")
                print("Veya: set_input ones (birlerle doldurmak için)")
                raise ValueError("Eksik parametre")

            # Özel durumlar için kontrol

            if parts[1] == "zeros":
                values = [0.0 for _ in range(len(layers[0]))]
                print("Giriş katmanı sıfırlarla sıfırlandı")
            elif parts[1] == "ones":
                values = [1.0 for _ in range(len(layers[0]))]
                print("Giriş katmanı birlerle dolduruldu")
            else:
                # Normal değer atama
                values = list(map(float, parts[1:]))
                if len(values) > len(layers[0]):
                    print(f"Uyarı: {len(values)} değer verildi ama sadece ilk {len(layers[0])} tanesi kullanılacak")

            # Değerleri giriş katmanına ata
            for i, val in enumerate(values[:len(layers[0])]):
                layers[0][i].value = val

            # Güncellenen değerleri göster
            print("\nGiriş Katmanı Değerleri:")
            for i in range(0, min(10, len(layers[0]))):  # İlk 10 değeri göster
                print(f"Nöron {i}: {layers[0][i].value:.4f}")
            if len(layers[0]) > 10:
                print(f"... (toplam {len(layers[0])} nöron)")

            # İstatistikler
            print(f"\nİstatistikler:")
            print(f"- Min: {min(values[:len(layers[0])]):.4f}")
            print(f"- Max: {max(values[:len(layers[0])]):.4f}")
            print(f"- Ortalama: {np.mean(values[:len(layers[0])]):.4f}")

        except Exception as e:
            print(f"Hata: {str(e)}")
            print("Doğru kullanım örnekleri:")
            print("- set_input 0.1 0.5 0.9 (belirli değerler atamak için)")
            print("- set_input random (rastgele değerler atamak için)")
            print("- set_input zeros (tüm girişleri sıfırlamak için)")
            print("- set_input ones (tüm girişleri 1 yapmak için)")
    
    # ... (diğer komutlar aynı şekilde kalabilir)

    else:
        print("\nKullanılabilir Komutlar:")
        print("- refresh(): Ağı yenile")
        print("- print_network(): Ağ yapısını terminalde göster")
        print("- get_connection(from_id,to_id): Bağlantı ağırlığını göster")
        #print("- draw(): Çizim modunu aç")
        #print("- add_neuron(layer_idx,value): Yeni nöron ekle")
        #print("- removeNeuron(id): Nöron sil")
        #print("- addLayer(index;[nöronlar]): Katman ekle")
        #print("- removeLayer(index): Katman sil")
        print("- changeW(from;to;weight): Ağırlık değiştir")
        print("- changeN(id;value): Nöron değeri değiştir")
        print("- visualize(): Ağ görselleştirmeyi aç/kapat")
        #print("- train_mnist([epochs;batch;lr;test_size;intel]): MNIST ile eğitim")
        #print("- train_digits([epochs;batch;lr;test_size;intel]): Digits ile eğitim")
        print("- train_custom(dosya.csv[;epochs;batch;lr;test_size;intel]): Özel veri ile eğitim")
        print("- set_input values:giriş değerlerini belirle")
        print("- bias(True or False) :bias değerlerini açıp kapatmaya yarıyor boş bırakılırsa açık mı kapalı mı onu veriyor")
        print("- history() :geçmişte yazılan komutları veriyor")
        print("- loadModel(dosyaAdı) : hazır modeli dosyadan yüklüyor")
        print("- exit: Programdan çık")

    cmd = input("\nKomut girin: ")
    if debug:
        if len(cmdHistory)>20:
            cmdHistory.pop(0)
        cmdHistory.append(cmd)