import csv
import time
import traceback
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
import math

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


import modeltrainingprogram 

import pyqtgraph as pg
import networkx as nx
import numpy as np
from pyqtgraph.Qt import QtGui , QtCore  # Add this import
from pyqtgraph.Qt import QtWidgets


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class Neuron:
    next_id = 0  # Global olarak artan ID değeri
    
    def __init__(self, default_value:float=0.0, activation_type='sigmoid'):
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
        elif self.activation_type == 'relu':
            return max(0, x)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def activation_derivative(self):
        if self.activation_type == 'sigmoid':
            # Çıktıyı 0.01-0.99 aralığına sıkıştır
            safe_output = np.clip(self.output, 0.01, 0.99)
            return safe_output * (1 - safe_output)  # f'(x) = f(x)(1 - f(x))
        elif self.activation_type == 'tanh':
            return 1 - self.output ** 2  # f'(x) = 1 - f(x)^2
        elif self.activation_type == 'relu':
            return 1 if self.weightedSum > 0 else 0  # ReLU türevi
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
debug = False  # Global debug değişkeni
#cmd = "train_custom(veri.csv;2,5,2;0.0004)" #program başlar başlamaz çalışacak ilk komut
cmd="train_digits(3)"


# Ağ oluşturma
randomMinWeight = -2.0
randomMaxWeight = 2.0

activation_types = ['sigmoid', 'tanh', 'relu']
defaultNeuronActivationType='relu'






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
    
    for neuronCount in neuronInLayers:

        # Her katman için yeni nöron listesi oluştur
        layer = [Neuron(default_value=1) for _ in range(neuronCount)]
        layers.append(layer)
    
    setConnections()



# Kullanım örneği:





"""
{
    0: {  # 1. katmandan 2. katmana
        0: [Connection(from_id=0, to_id=4, weight=0.45), 
            Connection(from_id=0, to_id=5, weight=-0.23),
            Connection(from_id=0, to_id=6, weight=0.14)],
        1: [Connection(from_id=1, to_id=4, weight=0.75),
            Connection(from_id=1, to_id=5, weight=-0.11),
            Connection(from_id=1, to_id=6, weight=0.32)],
        2: [Connection(from_id=2, to_id=4, weight=-0.87),
            Connection(from_id=2, to_id=5, weight=0.93),
            Connection(from_id=2, to_id=6, weight=0.56)]
    },
    1: {  # 2. katmandan 3. katmana
        4: [Connection(from_id=4, to_id=8, weight=0.21), 
            Connection(from_id=4, to_id=9, weight=-0.13)],
        5: [Connection(from_id=5, to_id=8, weight=0.68),
            Connection(from_id=5, to_id=9, weight=-0.50)],
        6: [Connection(from_id=6, to_id=8, weight=-0.91),
            Connection(from_id=6, to_id=9, weight=0.73)],
        7: [Connection(from_id=7, to_id=8, weight=0.34),
            Connection(from_id=7, to_id=9, weight=-0.66)]
    },
    2: {  # 3. katmandan 4. katmana
        8: [Connection(from_id=8, to_id=11, weight=-0.15),
            Connection(from_id=8, to_id=12, weight=0.42)],
        9: [Connection(from_id=9, to_id=11, weight=0.77),
            Connection(from_id=9, to_id=12, weight=-0.29)],
        10: [Connection(from_id=10, to_id=11, weight=-0.09),
            Connection(from_id=10, to_id=12, weight=0.89)]
    }
}

2. Bir Nöronun Bağlantılarına Erişim
Belirli bir nöronun bağlantılarına, nöronun ID'si ile ulaşabilirsiniz. Örneğin, 0. katmandaki 0. nöronun bağlantılarına şu şekilde erişebilirsiniz:

neuron_0_connections = connections[0][0]

Bu, 0. katmandaki ilk nöronun bağlantılarını içeren bir liste döndürecektir.

for conn in connections[0][0]:
    from_neuron_id = conn.connectedTo[0]  # Kaynak nöron ID'si
    to_neuron_id = conn.connectedTo[1]    # Hedef nöron ID'si
    print(f"Bağlantı Kaynak: {from_neuron_id}, Hedef: {to_neuron_id}, Ağırlık: {conn.weight}")


"""









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
start_time = None
enable_logging = True  # Loglama varsayılan olarak kapalı

def signal_handler(sig, frame):
    """Ctrl+C ile çıkış yakalandığında çağrılacak fonksiyon"""
    global enable_logging
    
    print("\nEğitim durduruldu.")
    
    if enable_logging:
        print("Veriler kaydediliyor...")
        
        visualize_saved_errors(save_and_plot_errors())
    
    exit(0)

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
        "final_error": error_history[-1] if error_history else None
    }
    
    with open(outputFolder+output_file_json, 'w') as f:
        json.dump(data, f)
    
    print(f"Hata verileri {output_file_json} dosyasına kaydedildi.")
    
    # Grafiği oluştur ve kaydet
    plt.figure(figsize=(12, 6))
    
    # Hata eğrisini çiz
    plt.plot(epoch_history, error_history, 'b-', linewidth=1)
    plt.plot(epoch_history, error_history, 'ro', markersize=3)
    
    # Grafiği biçimlendir
    plt.title('Eğitim Sırasında Ortalama Hata Değişimi')
    plt.xlabel('Epoch')
    plt.ylabel('Ortalama Hata')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Y eksenini logaritmik yap (opsiyonel - hatanın hızlı değişimlerini daha iyi gösterir)
    if min(error_history) > 0:  # Logaritmik eksen için tüm değerler pozitif olmalı
        plt.yscale('log')
    
    # Son hata değerini grafiğe ekle
    plt.annotate(f'Son Hata: {error_history[-1]:.6f}',
                xy=(epoch_history[-1], error_history[-1]),
                xytext=(max(0, epoch_history[-1]-1), error_history[-1]*1.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    # Grafiği kaydet
    plt.savefig(outputFolder+output_file_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Hata grafiği {output_file_png} dosyasına kaydedildi.")
    return outputFolder+output_file_json

def train_network(X_train, y_train, batch_size=1, epochs=None, intelligenceValue=None, learning_rate=0.05, output_graph=enable_logging):
    global error_history, epoch_history, start_time, enable_logging
    
    # Loglama ayarını güncelle
    #enable_logging = output_graph
    
    if enable_logging:
        print("Hata grafiği kaydı etkin. Eğitim sonunda veya Ctrl+C ile çıkış yaptığınızda grafik oluşturulacak.")
    
    # Hata geçmişi ve epoch geçmişini sıfırla
    error_history = []
    epoch_history = []
    
    # Ctrl+C sinyalini yakalamak için handler kaydet
    signal.signal(signal.SIGINT, signal_handler)
    
    # Kortikal kolon oluştur
    cortical_column = CorticalColumn()
    
    avg_error = float('inf')
    epoch = 0
    total_samples = len(X_train)
    start_time = time.time()
    last_print_time = start_time

    # Eğitim öncesi kontrol
    if len(layers[0]) != len(X_train[0]):
        print(f"Uyarı: Giriş boyutu uyumsuz! Ağ girişi: {len(layers[0])}, Veri girişi: {len(X_train[0])}")
        return

    try:
        # Eğitim döngüsü
        while True:

                
            total_error = 0
            processed_samples = 0
            epoch_gradients = []  # Gradyanları topla
            korteksChanges=[]
            
            # Batch işleme
            for batch_start in range(0, total_samples, batch_size):
                
                batch_end = min(batch_start + batch_size, total_samples)
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                batch_error = 0
                
                for X, y in zip(X_batch, y_batch):
                    cortical_column.backpropagation(X,y,learning_rate)
                    

                    # Hata hesapla
                    output = [neuron.value for neuron in layers[-1][:len(y)]]
                    error = hata_payi(y, output)
                    batch_error += error
                    
                    # Gradyanları hesapla

                    
                    gradients = compute_gradients(y, output)
                    gradient_dicts = [{'layer': layer_idx, 'grad': grad} for layer_idx, grad in gradients.items()]
                    epoch_gradients.extend(gradient_dicts)
                    recent_avg_error = cortical_column.monitor_network(avg_error)
                        
                    changes = cortical_column.adapt_neurons()
                    if changes:
                        korteksChanges.append({
                            'type': 'neuron_adaptation',
                            'epoch': epoch,
                            'batch': batch_start,
                            'changes': changes,
                            'before_error': avg_error
                        })
                    
                    conn_changes = cortical_column._adapt_connections()
                    if conn_changes:
                        korteksChanges.append({
                            'type': 'connection_adaptation',
                            'epoch': epoch,
                            'batch': batch_start,
                            'changes': conn_changes,
                            'before_error': avg_error
                        })
                    
                    

                    
                
                
                # Batch istatistikleri
                batch_error /= len(X_batch)
                total_error += batch_error * len(X_batch)
                processed_samples += len(X_batch)
                
                # Ortalama hatayı güncelle
                avg_error = total_error / processed_samples


                                # Debug logları
                if debug and korteksChanges:
                    print(f"\n[DEBUG] Son Değişiklikler:")
                    for change in korteksChanges[-min(3, len(korteksChanges)):]:  # Son 3 değişikliği göster
                        print(f" - {change['type']} at epoch {change['epoch']}, batch {change['batch']}")
                        print(f"   Error before: {change['before_error']:.6f}")
                        if 'changes' in change:
                            print(f"   Changes made: {len(change['changes'])}")
                

                    
                # İlerleme raporu
                current_time = time.time()
                if current_time - last_print_time > 10 or batch_end >= total_samples:
                    elapsed_time = current_time - start_time
                    samples_per_sec = processed_samples / elapsed_time if elapsed_time > 0 else 0
                    remaining_time = (total_samples - processed_samples) / samples_per_sec if samples_per_sec > 0 else 0
                    
                    print(f"\nEpoch {epoch+1}/{epochs if epochs else '∞'} - İlerleme: {processed_samples}/{total_samples} ({100*processed_samples/total_samples:.1f}%)")
                    print(f"Ortalama Hata: {avg_error:.6f}")
                    print(f"Geçen Süre: {elapsed_time/60:.1f} dak - Tahmini Kalan Süre: {remaining_time/60:.1f} dak")
                    print(f"Örnek/Saniye: {samples_per_sec:.1f}")
                    
                    # Eğer loglama etkinse, hata değerini kaydet (ara dosya oluşturmadan)
                    if enable_logging:
                        error_history.append(avg_error)
                        epoch_history.append(epoch + processed_samples/total_samples)
                    
                    last_print_time = current_time
                                    # IntelligenceValue kontrolü (her batch sonunda)
            if epochs > 1 and epoch >= epochs:
                break
            if epochs <1 and epochs>avg_error:
                print(f"\nHata {epochs} değerinin altına düştü! Eğitim durduruldu.")
                break
            
            # Epoch sonu işlemleri
            epoch += 1
            
        

            

        
        # Final raporu
        total_time = time.time() - start_time
        print(f"\n=== EĞİTİM TAMAMLANDI ===")
        print(f"Toplam Değişiklik: {len(korteksChanges)}")
        print(f"Toplam Süre: {total_time/60:.1f} dakika")
        print(f"Son Hata: {avg_error:.6f}")
        print(f"Toplam Epoch: {epoch}")
        print(f"Final Ağ Yapısı: {[len(layer) for layer in layers]}")

        
        # Eğitim tamamlandığında hata verilerini kaydet ve görselleştir (eğer loglama etkinse)
        if enable_logging:
            visualize_saved_errors(save_and_plot_errors())
        
    except KeyboardInterrupt:
        # Ctrl+C ile çıkış yakalandı, signal handler zaten işleyecek
        pass
    
    return cortical_column, avg_error




import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def visualize_saved_errors(filename):
    """Kaydedilmiş hata verilerini gelişmiş grafiklerle görselleştir"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    errors = np.array(data["errors"])
    epochs = np.array(data["epochs"])
    
    # Ana grafik
    plt.figure(figsize=(15, 10))
    
    # 1. Hata eğrisi (ana grafik)
    plt.subplot(2, 2, (1, 3))  # 2 satır, 2 sütun, 1 ve 3'ü birleştir
    main_plot = plt.plot(epochs, errors, 'b-', linewidth=1, label='Ortalama Hata')
    plt.plot(epochs, errors, 'ro', markersize=1)
    
    # Eğilim çizgisi ekleme
    z = np.polyfit(epochs, errors, 3)
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), "r--", linewidth=2, label='Eğilim Çizgisi')
    
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
    plt.subplot(2, 2, 2)
    last_20 = int(len(epochs) * 0.8)
    plt.plot(epochs[last_20:], errors[last_20:], 'b-', linewidth=1.5)
    plt.plot(epochs[last_20:], errors[last_20:], 'ro', markersize=2)
    
    # Son bölüm için lineer regresyon
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        epochs[last_20:], errors[last_20:])
    plt.plot(epochs[last_20:], intercept + slope*epochs[last_20:], 
             'g--', linewidth=2, 
             label=f'Eğim: {slope:.2e}\nR²: {r_value**2:.2f}')
    
    plt.title('Son %20 Epoch Yakınlaştırma')
    plt.xlabel('Epoch')
    plt.ylabel('Hata')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Hata dağılımı histogramı
    plt.subplot(2, 2, 4)
    plt.hist(errors, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Hata Dağılımı')
    plt.xlabel('Hata Değeri')
    plt.ylabel('Frekans')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Genel bilgiler
    stats_text = (
        f"Başlangıç Hata: {errors[0]:.6f}\n"
        f"Son Hata: {errors[-1]:.6f}\n"
        f"En Düşük Hata: {np.min(errors):.6f}\n"
        f"Ortalama Hata: {np.mean(errors):.6f}\n"
        f"Standart Sapma: {np.std(errors):.6f}\n"
        f"Dönüm Noktaları: {len(turning_points)}\n"
        f"Toplam Epoch: {len(epochs)}\n"
        f"Toplam Süre: {data['total_time_seconds']:.2f} sn"
    )
    
    plt.figtext(0.75, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.5), 
                fontsize=9)
    
    plt.tight_layout()
    
    # Grafiği kaydet
    output_file = os.path.splitext(filename)[0] + "_advanced_viz.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gelişmiş hata grafiği {output_file} dosyasına kaydedildi.")
    
    # Eğilim analizi
    analyze_trend(errors, epochs)

def analyze_trend(errors, epochs):
    """Hata eğilimini analiz eder ve yorumlar"""
    # Son %20'lik kısım için eğim analizi
    last_20 = int(len(epochs) * 0.8)
    slope, _, _, _, _ = stats.linregress(epochs[last_20:], errors[last_20:])
    
    print("\n=== HATA EĞİLİM ANALİZİ ===")
    print(f"Son hata değeri: {errors[-1]:.6f}")
    print(f"Son %20 epoch'taki ortalama hata eğimi: {slope:.2e}")
    
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







# Önceki import'ların yanına ekleyin
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton


# Global değişkenler
current_dataset = None  # 'mnist' veya 'digits' olabilir
drawing_mode = False
drawing_canvas = None
drawing_data = None  # Veri setine göre boyutlandırılacak

def toggle_drawing_mode(event):
    global drawing_mode
    drawing_mode = not drawing_mode
    if drawing_mode:
        print("Çizim modu aktif! Sol tıkla çiz, sağ tıkla temizle, 'p' ile tahmin et.")
    else:
        print("Çizim modu pasif.")

def handle_drawing_click(event):
    """Fare tıklamasını işler - veri setine göre uyarlanmış"""
    if not drawing_mode or event.inaxes != ax_drawing:
        return
    
    # Piksel boyutlarını al
    rows, cols = drawing_data.shape
    
    if event.button == MouseButton.LEFT:
        # Piksel koordinatlarını al
        x, y = int(event.xdata), int(event.ydata)
        
        # Veri setine göre boyama alanını ayarla
        brush_size = 3 if current_dataset == 'mnist' else 0.2
        
        # Belirlenen alanı boya
        for i in range(-brush_size//2, brush_size//2 + 1):
            for j in range(-brush_size//2, brush_size//2 + 1):
                if 0 <= x+i < cols and 0 <= y+j < rows:
                    drawing_data[y+j, x+i] = min(1.0, drawing_data[y+j, x+i] + 0.3)
        
        update_drawing_display()
    
    elif event.button == MouseButton.RIGHT:
        # Temizle
        drawing_data.fill(0)
        update_drawing_display()

def handle_drawing_motion(event):
    if not drawing_mode or event.button != MouseButton.LEFT or event.inaxes != ax_drawing:
        return
    
    # Fare hareket ederken çizim yap
    x, y = int(event.xdata), int(event.ydata)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= x+i < 28 and 0 <= y+j < 28:
                drawing_data[y+j, x+i] = min(1.0, drawing_data[y+j, x+i] + 0.3)
    
    update_drawing_display()

def update_drawing_display():
    global drawing_canvas
    if drawing_canvas:
        drawing_canvas.set_data(drawing_data)
        plt.draw()

def predict_drawing(event):
    """Çizimi tahmin et - veri setine göre uyarlanmış"""
    if not drawing_mode:
        return
    
    # Çizimi ağa uygun formata getir
    flat_drawing = drawing_data.flatten()
    
    # Giriş katmanını doldur
    for i in range(min(len(layers[0]), len(flat_drawing))):
        layers[0][i].value = flat_drawing[i]
    
    # Tahmin yap
    runAI()
    
    # Sonuçları göster
    output_layer = layers[-1]
    predicted_digit = np.argmax([neuron.value for neuron in output_layer])
    confidence = output_layer[predicted_digit].value * 100
    
    print(f"\nTahmin: {predicted_digit} (%{confidence:.2f} güven)")
    print("Detaylı çıktılar:")
    for i, neuron in enumerate(output_layer):
        print(f"{i}: %{neuron.value*100:.2f}")
    
    # Çizimi göster (MNIST için büyütülmüş hal)
    if current_dataset == 'mnist':
        plt.figure()
        plt.imshow(drawing_data, cmap='gray_r')
        plt.title(f"Tahmin: {predicted_digit} (%{confidence:.2f})")
        plt.axis('off')
        plt.show()
    else:
        # Digits için daha küçük gösterim
        plt.figure(figsize=(2,2))
        plt.imshow(drawing_data, cmap='gray_r')
        plt.title(f"{predicted_digit}")
        plt.axis('off')
        plt.show()

def setup_drawing_area(dataset_type='mnist'):
    """Veri setine göre çizim alanını ayarlar"""
    global ax_drawing, drawing_canvas, drawing_data, current_dataset, brush_size
    
    current_dataset = dataset_type
    
    # Veri setine göre boyutları ayarla
    if dataset_type == 'mnist':
        drawing_data = np.zeros((28, 28))
        brush_size = 3  # MNIST için orta boy fırça
    else:  # digits
        drawing_data = np.zeros((8, 8))
        brush_size = 0.2  # Digits için küçük fırça
    
    fig = plt.figure(figsize=(10, 5) if dataset_type == 'mnist' else (8, 4))
    ax_drawing = fig.add_subplot(121)
    ax_drawing.set_title(f"{dataset_type.upper()} Rakam Çiz (Sol tık: çiz, Sağ tık: temizle)")
    ax_drawing.set_xticks([])
    ax_drawing.set_yticks([])
    
    drawing_canvas = ax_drawing.imshow(drawing_data, cmap='gray_r', vmin=0, vmax=1)
    
    # Butonlar
    ax_button = fig.add_subplot(222)
    btn_toggle = Button(ax_button, 'Çizim Modu Aç/Kapat')
    btn_toggle.on_clicked(toggle_drawing_mode)
    
    ax_predict = fig.add_subplot(224)
    btn_predict = Button(ax_predict, 'Tahmin Et (p)')
    btn_predict.on_clicked(predict_drawing)
    
    # Klavye ve fare olayları
    fig.canvas.mpl_connect('button_press_event', handle_drawing_click)
    fig.canvas.mpl_connect('motion_notify_event', handle_drawing_motion)
    fig.canvas.mpl_connect('key_press_event', lambda e: predict_drawing(e) if e.key == 'p' else None)
    
    plt.tight_layout()
    plt.show()






import struct
import gzip

def read_mnist_images(filename):
    """MNIST .idx3-ubyte dosyasını okur ve numpy array olarak döndürür"""
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return images / 255.0  # 0-1 aralığına normalize et

def read_mnist_labels(filename):
    """MNIST .idx1-ubyte dosyasını okur ve numpy array olarak döndürür"""
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def prepare_mnist_data(images_file, labels_file=None):
    """MNIST verilerini hazırlar"""
    X = read_mnist_images(images_file)
    
    if labels_file:
        y_labels = read_mnist_labels(labels_file)
        # One-hot encoding yap
        y = np.zeros((len(y_labels), 10))
        y[np.arange(len(y_labels)), y_labels] = 1
        return X, y
    return X


def train_mnist(images_file, labels_file, epochs=10, batch_size=256, learning_rate=0.1, test_size=0.2, intelligence=None):
    """MNIST veri setini eğitir"""
    global current_dataset, layers, connections
    current_dataset = 'mnist'
    


    
    # Verileri yükle
    X_train, y_train = prepare_mnist_data(images_file, labels_file)
    
    # Eğitim
    train_network(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        intelligenceValue=intelligence
    )
    
    # Test
    X_test, y_test = prepare_mnist_data(
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    )
    evaluate_network(X_test, y_test)

def train_digits(epochs=10, batch_size=64, learning_rate=0.1, test_size=0.2, intelligence=None):
    """Digits veri setini eğitir"""
    global current_dataset, layers, connections
    current_dataset = 'digits'
    
    # Veri setini yükle
    digits = load_digits()
    X = digits.data / 16.0  # Normalize
    y = np.zeros((len(digits.target), 10))
    y[np.arange(len(digits.target)), digits.target] = 1  # One-hot encoding
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    

    

    
    # Eğitim
    train_network(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        intelligenceValue=intelligence
    )
    
    # Test
    evaluate_network(X_test, y_test)

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




def disable_all_biases():
    for layer_idx in connections:
        for neuron_id in connections[layer_idx]:
            for conn in connections[layer_idx][neuron_id]:
                conn.bias = 0

# Kullanım:
disable_all_biases()

def enable_all_biases():
    for layer_idx in connections:
        for neuron_id in connections[layer_idx]:
            for conn in connections[layer_idx][neuron_id]:
                conn.bias = random.uniform(-0.1, 0.1)  # Rastgele küçük değerlerle yeniden başlat

# Kullanım:
#enable_all_biases()





class CorticalColumn:
    """
    [Başla]  
    │    
    ├─► Backpropagation Çalıştır
    ├─► Nöron Sağlığını Hesapla  
    ├─► Kritik Nöronları Değiştir (Birleştir/Böl/Sil)
    ├─► Katman Sağlığını Ölç  
    ├─► Katman Birleştir/Sil (Gerekirse)  
    ├─► Değişiklikleri Kaydet  
    │  
    └─ Hata Arttı mı? ──Evet─► Geri Al  
           │  
           Hayır  
           │  
          [Devam]  """
    
    """
    Değişiklik seçenekleri:
    -Nöron değişiklikleri yapıldı:
        +Nöronlar birleştirildi{[id0,id1...],birleştirilmeden önce nöronların bağlantıları ve bağlnaıt ağırlıkları{id0:conn(weight,connectedToArg=[]),id1:...}}
        +Nöron bölündü{[id0]->[id0,id1...],bölünmeden önce nöronun bağlantıları ve bağlnaıt ağırlıkları{id0:conn(weight,connectedToArg=[])}}
        +Nöronlar silindi{id0,silinmeden önceki bağlantılar ve bağlantı ağırlıkları}
    -Katman silindi{layerid,katmandaki nöronlar ve o nöronların bağlantıları ve o bağlantıların ağırlıkları}

    
    
    
    """
    def __init__(self):
        global layers,connections
        self.neuronHealtThreshould = 0.3

    def monitor_network(self,avgErrorArg):
        global layers,connections


    def backpropagation(self,input_data, target_data, learning_rate=0.1):
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
                            weight_change = learning_rate * deltas[next_layer_idx][j] * neuron.value
                            # Mevcut Connection sınıfınızdaki update_weight metodunu kullan
                            conn.update_weight(learning_rate, deltas[next_layer_idx][j] * neuron.value)
                            break
                        
        # Toplam hata değerini hesapla ve döndür (MSE - Mean Squared Error)
        total_error = sum(error**2 for error in output_errors) / len(output_errors) if output_errors else 0
        return total_error


    def adapt_neurons(self):
        global layers,connections
        weakNeurons=[]
        # Tüm nöronların sağlığını hesapla
        for layer_idx, layer in enumerate(layers):
            
            for neuron_idx, neuron in enumerate(layer):
                neuron,health = self.calculate_neuron_health(neuron)
                if health < self.neuronHealtThreshould:
                    weakNeurons.append(neuron)
                if debug :print(f"  Nöron {neuron.id} sağlık puanı: {health:.4f}")
    def _adapt_connections(self):
        global layers,connections
    
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

        return neuron,health_score
        
    def find_neuron_layer(self,neuron_id):
        """Verilen ID'ye sahip nöronun hangi katmanda olduğunu bulur"""
        global layers
        for layer_idx, layer in enumerate(layers):
            for neuron in layer:
                if neuron.id == neuron_id:
                    return layer_idx
        return None

    def add_neuron_to_layer(self,layer_index=None, neuron_id=None, activation_type=defaultNeuronActivationType):
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
        new_neuron = Neuron(activation_type=activation_type)

        # Yeni nöronu katmana ekle
        layers[layer_index].append(new_neuron)

        # Önceki katman varsa, önceki katmandan bu nörona bağlantılar oluştur
        if layer_index > 0:
            prev_layer_idx = layer_index - 1
            for prev_neuron in layers[prev_layer_idx]:
                weight = random.uniform(randomMinWeight, randomMaxWeight)
                conn = Connection(connectedToArg=[prev_neuron.id, new_neuron.id], weight=weight)

                if prev_neuron.id not in connections[prev_layer_idx]:
                    connections[prev_layer_idx][prev_neuron.id] = []

                connections[prev_layer_idx][prev_neuron.id].append(conn)

        # Sonraki katman varsa, bu nörondan sonraki katmana bağlantılar oluştur
        if layer_index < len(layers) - 1:
            for next_neuron in layers[layer_index + 1]:
                weight = random.uniform(randomMinWeight, randomMaxWeight)
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




# Terminal giriş döngüsü - Dinamik versiyon
while True:
    if cmd == "exit":
        break
        
    if cmd == "refresh()":
        clearGUI()
        runAI()
        visualize_network(layers, connections, refresh=True)
        print(getOutput())
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

    
    elif cmd == "draw()":
        if current_dataset is None:
            print("Önce bir veri seti ile eğitim yapın (train_mnist veya train_digits)")
        else:
            setup_drawing_area(current_dataset)
    
    elif cmd == "visualize()":
        visualizeNetwork = not visualizeNetwork
        print(f"Görselleştirme {'aktif' if visualizeNetwork else 'pasif'}")

    elif cmd.startswith("train_mnist("):
        try:
            # Parametreleri ayrıştır (varsayılan değerlerle)
            params = cmd[len("train_mnist("):-1].split(";")
            
            # Varsayılan değerler
            epochs = 10
            batch_size = 256
            learning_rate = 0.1
            test_size = 0.2
            intelligence = None
            
            # Kullanıcı parametrelerini al
            if len(params) > 0 and params[0].strip(): epochs = int(params[0])
            if len(params) > 1 and params[1].strip(): batch_size = int(params[1])
            if len(params) > 2 and params[2].strip(): learning_rate = float(params[2])
            if len(params) > 3 and params[3].strip(): test_size = float(params[3])
            if len(params) > 4 and params[4].strip(): intelligence = float(params[4])
            
            print(f"\nMNIST Eğitim Parametreleri:")
            print(f"- Epochs: {epochs}")
            print(f"- Batch Size: {batch_size}")
            print(f"- Learning Rate: {learning_rate}")
            print(f"- Test Size: {test_size}")
            print(f"- Intelligence Threshold: {intelligence if intelligence is not None else 'Kapalı'}")
            
            train_mnist(
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                test_size=test_size,
                intelligence=intelligence
            )
            
        except Exception as e:
            print("Hatalı komut formatı! Örnek kullanımlar:")
            print("train_mnist()  # Tüm varsayılan değerlerle")
            print("train_mnist(5)  # Sadece epoch belirterek")
            print("train_mnist(10;128;0.05;0.1;0.01)  # Tüm parametrelerle")
            print("Parametre sırası: epochs;batch_size;learning_rate;test_size;intelligence")
            traceback.print_exc()

    elif cmd.startswith("train_digits("):
        try:
            setLayers([64,32,10])
            # Parametreleri ayrıştır (varsayılan değerlerle)
            params = cmd[len("train_digits("):-1].split(";")
            
            # Varsayılan değerler
            epochs = 10
            batch_size = 64
            learning_rate = 0.1
            test_size = 0.2
            intelligence = None
            
            # Kullanıcı parametrelerini al
            if len(params) > 0 and params[0].strip(): epochs = int(params[0])
            if len(params) > 1 and params[1].strip(): batch_size = int(params[1])
            if len(params) > 2 and params[2].strip(): learning_rate = float(params[2])
            if len(params) > 3 and params[3].strip(): test_size = float(params[3])
            if len(params) > 4 and params[4].strip(): intelligence = float(params[4])
            
            print(f"\nDigits Eğitim Parametreleri:")
            print(f"- Epochs: {epochs}")
            print(f"- Batch Size: {batch_size}")
            print(f"- Learning Rate: {learning_rate}")
            print(f"- Test Size: {test_size}")
            print(f"- Intelligence Threshold: {intelligence if intelligence is not None else 'Kapalı'}")
            
            train_digits(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                test_size=test_size,
                intelligence=intelligence
            )
            
        except Exception as e:
            print("Hatalı komut formatı! Örnek kullanımlar:")
            print("train_digits()  # Tüm varsayılan değerlerle")
            print("train_digits(20)  # Sadece epoch belirterek")
            print("train_digits(15;32;0.01;0.3;0.005)  # Tüm parametrelerle")
            print("Parametre sırası: epochs;batch_size;learning_rate;test_size;intelligence")
            traceback.print_exc()

    elif cmd.startswith("train_custom("):
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
            if parts[1] == "random":
                values = [random.uniform(0.1, 1.0) for _ in range(len(layers[0]))]
                print(f"Giriş katmanı rastgele değerlerle güncellendi (0.1-1.0 aralığı)")
            elif parts[1] == "zeros":
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
        print("- draw(): Çizim modunu aç")
        print("- add_neuron(layer_idx,value): Yeni nöron ekle")
        print("- removeNeuron(id): Nöron sil")
        print("- addLayer(index;[nöronlar]): Katman ekle")
        print("- removeLayer(index): Katman sil")
        print("- changeW(from;to;weight): Ağırlık değiştir")
        print("- changeN(id;value): Nöron değeri değiştir")
        print("- visualize(): Ağ görselleştirmeyi aç/kapat")
        print("- train_mnist([epochs;batch;lr;test_size;intel]): MNIST ile eğitim")
        print("- train_digits([epochs;batch;lr;test_size;intel]): Digits ile eğitim")
        print("- train_custom(dosya.csv[;epochs;batch;lr;test_size;intel]): Özel veri ile eğitim")
        print("- set_input values:giriş değerlerini belirle")
        print("- exit: Programdan çık")

    cmd = input("\nKomut girin: ")