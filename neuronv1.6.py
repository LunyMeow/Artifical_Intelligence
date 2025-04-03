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
    
    def __init__(self, default_value=0.0, activation_type='sigmoid'):
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
            return 1 if self.output > 0 else 0  # ReLU türevi
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
    def __init__(self, weight=0, fromTo=[0, 0], bias=0.1):  # Varsayılan bias=0.1
        self.weight = weight
        self.connectedTo = fromTo
        self.bias = bias  # Bias parametresi eklendi
    
    def update_weight(self, learning_rate, delta):
        self.weight += learning_rate * delta
        self.bias += learning_rate * delta * 0.1  # Bias da güncelleniyor



visualizeNetwork =False
debug = False  # Global debug değişkeni

# Ağ oluşturma
randomMinWeight = -5.0
randomMaxWeight = 5.0

activation_types = ['sigmoid', 'tanh', 'relu']
defaultNeuronActivationType='relu'




# Ağ Yapısı
input_size = 2  # 28x28 piksel girişi
hidden_size = 4  # Gizli katmanda 128 nöron
output_size = 1  # 0-9 için 10 çıktı nöronu

# Önce boş bir layers listesi oluştur
layers = []



# Bağlantıları oluşturma

connections = {}
def setConnections():
    global layers,connections
    connections.clear()
    connections = {layer_idx: {} for layer_idx in range(len(layers) - 1)}
    for layer_idx in range(len(layers) - 1):
        for neuron in layers[layer_idx]:
            for next_neuron in layers[layer_idx + 1]:
                conn = Connection(fromTo=[neuron.id, next_neuron.id], # Bağlantıları oluştururken daha akıllı ilkleme yapın
    weight = random.uniform(-1/np.sqrt(len(layers[layer_idx])), 
                          1/np.sqrt(len(layers[layer_idx])))) #weight=random.uniform(randomMinWeight, randomMaxWeight)
                if neuron.id not in connections[layer_idx]:
                    connections[layer_idx][neuron.id] = []
                connections[layer_idx][neuron.id].append(conn)

def setLayers(*neuronInLayers):
    """Katmanları ve nöron sayılarını ayarlar"""
    global layers  # Global layers listesini kullanacağımızı belirtiyoruz
    layers.clear()  # Önceki katmanları temizle
    
    for neuronCount in neuronInLayers:
        # Her katman için yeni nöron listesi oluştur
        layer = [Neuron(1) for _ in range(neuronCount)]
        layers.append(layer)
    
    setConnections()

setLayers(2, 4, 1)  # 2 giriş, 4 gizli, 1 çıkış nöronu


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
    from_neuron_id = conn.fromTo[0]  # Kaynak nöron ID'si
    to_neuron_id = conn.fromTo[1]    # Hedef nöron ID'si
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

cmd = "train_custom(veri.csv;2,3,5,2;0.2)" #program başlar başlamaz çalışacak ilk komut




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


def removeLayer(layers, connections, layer_idx):
    """
    Belirtilen indeksteki katmanı ve ilgili bağlantıları siler.
    
    :param layers: Nöron katmanları
    :param connections: Katmanlar arası bağlantılar
    :param layer_idx: Silinecek katman indeksi
    """
    if layer_idx < 0 or layer_idx >= len(layers):
        print(f"Hata: Geçersiz katman indeksi {layer_idx}")
        return
    
    if len(layers) <= 2:
        print("Hata: En az bir giriş ve bir çıkış katmanı kalmalıdır")
        return
    
    # Katmanı sil
    deleted_layer = layers.pop(layer_idx)
    
    # Bağlantıları güncelle
    new_connections = {}
    
    # Önceki bağlantıları kopyala (indeksler değişecek)
    for i in connections:
        if i < layer_idx - 1:
            new_connections[i] = connections[i]
        elif i > layer_idx:
            new_connections[i-1] = connections[i]
    
    # Eğer silinen katman ilk veya son katman değilse, yeni bağlantı oluştur
    if 0 < layer_idx < len(layers):
        # Önceki katmandan sonraki katmana doğrudan bağlantı kur
        new_connections[layer_idx-1] = {}
        for prev_neuron in layers[layer_idx-1]:
            for next_neuron in layers[layer_idx]:
                conn = Connection(fromTo=[prev_neuron.id, next_neuron.id], 
                                weight=random.uniform(randomMinWeight, randomMaxWeight))
                if prev_neuron.id not in new_connections[layer_idx-1]:
                    new_connections[layer_idx-1][prev_neuron.id] = []
                new_connections[layer_idx-1][prev_neuron.id].append(conn)
    
    connections.clear()
    connections.update(new_connections)
    
    print(f"Katman {layer_idx} ve bağlantıları silindi. {len(deleted_layer)} nöron kaldırıldı.")




def removeNeuron(layers, connections, neuron_id):
    # Nöronu bul
    neuron = None
    layer_idx = -1
    for i, layer in enumerate(layers):
        for n in layer:
            if n.id == neuron_id:
                neuron = n
                layer_idx = i
                break
        if neuron:
            break
    
    if not neuron:
        print(f"Hata: {neuron_id} ID'li nöron bulunamadı")
        return
    
    # Nöronu katmandan sil
    layers[layer_idx].remove(neuron)
    
    # Gelen bağlantıları sil (önceki katmandan)
    if layer_idx > 0:
        for prev_id in list(connections[layer_idx - 1].keys()):
            connections[layer_idx - 1][prev_id] = [
                conn for conn in connections[layer_idx - 1][prev_id] 
                if conn.connectedTo[1] != neuron_id
            ]
            if not connections[layer_idx - 1][prev_id]:
                del connections[layer_idx - 1][prev_id]
    
    # Giden bağlantıları sil (sonraki katmana)
    if layer_idx < len(layers) - 1 and layer_idx in connections:
        if neuron_id in connections[layer_idx]:
            del connections[layer_idx][neuron_id]



def add_neuron(layers, connections, layer_idx, new_neuron):
    layers[layer_idx].append(new_neuron)
    
    # Önceki katmandan bağlantıları oluştur
    if layer_idx > 0:
        for prev_neuron in layers[layer_idx - 1]:
            conn = Connection(fromTo=[prev_neuron.id, new_neuron.id], 
                            weight=random.uniform(randomMinWeight, randomMaxWeight))
            if prev_neuron.id not in connections[layer_idx - 1]:
                connections[layer_idx - 1][prev_neuron.id] = []
            connections[layer_idx - 1][prev_neuron.id].append(conn)
    
    # Sonraki katmana bağlantıları oluştur
    if layer_idx < len(layers) - 1:
        for next_neuron in layers[layer_idx + 1]:
            conn = Connection(fromTo=[new_neuron.id, next_neuron.id], 
                            weight=random.uniform(randomMinWeight, randomMaxWeight))
            if new_neuron.id not in connections[layer_idx]:
                connections[layer_idx][new_neuron.id] = []
            connections[layer_idx][new_neuron.id].append(conn)





def addLayer(layers, connections, layer_idx, neurons):
    layers.insert(layer_idx, neurons)
    
    # Bağlantıları güncelle (indeks kaydırma)
    new_connections = {}
    for i in connections:
        if i >= layer_idx:
            new_connections[i + 1] = connections[i]
        else:
            new_connections[i] = connections[i]
    connections.clear()
    connections.update(new_connections)
    
    # Yeni bağlantıları oluştur
    if layer_idx > 0:
        connections[layer_idx - 1] = {}
        for prev_neuron in layers[layer_idx - 1]:
            for new_neuron in neurons:
                conn = Connection(fromTo=[prev_neuron.id, new_neuron.id], 
                                weight=random.uniform(randomMinWeight, randomMaxWeight))
                if prev_neuron.id not in connections[layer_idx - 1]:
                    connections[layer_idx - 1][prev_neuron.id] = []
                connections[layer_idx - 1][prev_neuron.id].append(conn)
    
    if layer_idx < len(layers) - 1:
        connections[layer_idx] = {}
        for new_neuron in neurons:
            for next_neuron in layers[layer_idx + 1]:
                conn = Connection(fromTo=[new_neuron.id, next_neuron.id], 
                                weight=random.uniform(randomMinWeight, randomMaxWeight))
                if new_neuron.id not in connections[layer_idx]:
                    connections[layer_idx][new_neuron.id] = []
                connections[layer_idx][new_neuron.id].append(conn)



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



def calculate_neuron_delta(neuron, target_value):
    """Nöronun delta (hata) değerini hesaplar"""
    error = target_value - neuron.value
    derivative = neuron.activation_derivative()
    return error * derivative
    """# Son katman nöronu için delta hesapla
    output_neuron = layers[-1][0]  # Son katmanın ilk nöronu
    target = 0.8  # Beklenen çıktı
    delta = calculate_neuron_delta(output_neuron, target)"""

def calculate_weight_update(neuron, connection, target, learning_rate=1):
    delta = calculate_neuron_delta(neuron, target)
    from_neuron = get_neuron_by_id(connection.connectedTo[0])
    
    # Ağırlık ve bias güncellemesi
    weight_update = learning_rate * delta * from_neuron.value
    bias_update = learning_rate * delta * 0.1  # Bias için daha küçük adım
    
    return weight_update, bias_update  # Artık tuple döndürüyor


def diagnose_connection(neuron, connection,target, threshold=0.5):
    """Bağlantı sorunlarını teşhis eder"""
    from_neuron = get_neuron_by_id(connection.connectedTo[0])
    
    # 1. Aşırı değer kontrolü
    if abs(from_neuron.value) > 10:  # Aşırı aktivasyon
        return "Önceki nöron aşırı değer üretiyor: {from_neuron.value:.4f}"
    
    # 2. Ölü nöron kontrolü
    if abs(from_neuron.value) < 0.0001 and abs(connection.weight) < 0.0001:
        return "Ölü bağlantı (aktivasyon yok)"
    
    # 3. Tutarsız gradiyent kontrolü
    delta = calculate_neuron_delta(neuron, target)
    expected_grad = delta * from_neuron.value
    if abs(expected_grad) < threshold * 0.01:
        return "Zayıf gradiyent (öğrenme yok)"
    
    return "Bağlantı normal"

def connection_health_score(neuron, connection , target):
    """0-1 arasında bağlantı sağlık skoru"""
    from_neuron = get_neuron_by_id(connection.connectedTo[0])
    delta = calculate_neuron_delta(neuron, target)
    
    # 1. Aktivasyon uygunluğu (0-1 arasında olmalı)
    activation_score = 1 - min(abs(from_neuron.value - 0.5), 0.5)/0.5
    
    # 2. Gradiyent büyüklüğü
    grad = delta * from_neuron.value
    grad_score = min(abs(grad)*10, 1)
    
    # 3. Ağırlık büyüklüğü (aşırı büyük/küçük olmamalı)
    weight_score = 1 - min(abs(connection.weight)/5, 1)
    
    return (activation_score + grad_score + weight_score) / 3

def calculate_hidden_delta(neuron, next_layer_deltas):
    """Gizli katman nöronları için delta hesaplama"""
    # Sonraki katmandaki deltalar ve bağlantı ağırlıklarını topla
    weighted_sum = 0
    for conn in get_neuron_connections(neuron.id, outgoing=True):
        next_neuron_id = conn.connectedTo[1]
        next_neuron = get_neuron_by_id(next_neuron_id)
        next_delta = next_layer_deltas.get(next_neuron_id, 0)
        weighted_sum += conn.weight * next_delta
    
    return weighted_sum * neuron.activation_derivative()
def update_hidden_weights(layer_idx,target, learning_rate=0.01):
    """Gizli katman ağırlıklarını günceller"""
    for neuron in layers[layer_idx]:
        # Sonraki katman deltalarını topla
        next_layer_deltas = {}
        for next_neuron in layers[layer_idx+1]:
            next_layer_deltas[next_neuron.id] = calculate_neuron_delta(next_neuron, target)
        
        # Nöronun delta değerini hesapla
        delta = calculate_hidden_delta(neuron, next_layer_deltas)
        
        # Gelen bağlantıları güncelle
        for conn in get_neuron_connections(neuron.id, incoming=True):
            from_neuron = get_neuron_by_id(conn.connectedTo[0])
            weight_update = learning_rate * delta * from_neuron.value
            conn[2] += weight_update  # Ağırlığı güncelle


def adapt_network(currentError, target: list, perfectValueArg=0.3,learning_rateArg = 1,current_epoch=None, total_epochs=None):
    error = hata_payi(target, runAI())
    print(f"Hata değeri: {error}")
    # Add these at the beginning
    MAX_WEIGHT = 10.0
    GRADIENT_CLIP = 1.0
    gradients=[]

    def normalize_weight(w):
        return max(-MAX_WEIGHT, min(MAX_WEIGHT, w))
    
    def clip_gradient(g):
        return max(-GRADIENT_CLIP, min(GRADIENT_CLIP, g))

    for layer_idx, layer in enumerate(layers[::-1]):  # Katmanları ters sırada dolaş (çıkıştan girişe)
        current_layer_idx = len(layers) - 1 - layer_idx  # Gerçek katman indeksi
        is_last_layer = (current_layer_idx == len(layers) - 1)
        is_first_layer = (current_layer_idx == 0)

        if debug :
            print(f"\n=== Katman {current_layer_idx} {'(SON KATMAN)' if is_last_layer else '(GİZLİ KATMAN)' if not is_first_layer else '(GİRİŞ KATMANI)'} ===")

        for neuron_idx, neuron in enumerate(layer):
            if debug :
                print(f"\nNöron ID: {neuron.id} (Index: {neuron_idx})")
            neuronsConnections = get_neuron_connections(neuron.id, True, True)

            for conn in neuronsConnections:
                from_id = conn.connectedTo[0]
                to_id = conn.connectedTo[1]
                weight = conn.weight
                conn_layer_idx = current_layer_idx if to_id == neuron.id else current_layer_idx - 1
                
                grad_info={
                    'layer':current_layer_idx,
                    'from':conn.connectedTo[0],
                    'to':conn.connectedTo[1],
                    'grad':0.0,
                    'weight':conn.weight
                }

                # Bağlantı yönünü belirle
                if to_id == neuron.id:  # Gelen bağlantı
                    direction = f"{from_id} <- {to_id}"
                    connection_type = "GELEN"
                    from_neuron = get_neuron_by_id(from_id)
                    to_neuron = neuron
                else:  # Giden bağlantı
                    direction = f"{from_id} -> {to_id}"
                    connection_type = "GİDEN"
                    from_neuron = neuron
                    to_neuron = get_neuron_by_id(to_id)

                # Hesaplamalar
                if is_last_layer:
                    # SON KATMAN için işlemler
                    target_value = target[neuron_idx] if neuron_idx < len(target) else 0
                    delta = calculate_neuron_delta(neuron, target_value)

                    if to_id == neuron.id:  # Gelen bağlantı ise
                        # Modify weight update sections like this:
                        weight_update, bias_update = calculate_weight_update(neuron, conn, target_value)
                        weight_update = clip_gradient(weight_update * learning_rateArg)  # Apply learning rate and clip
                        
                        grad_info['grad'] = weight_update

                        new_weight = normalize_weight(conn.weight + weight_update)


                        diagnose = diagnose_connection(neuron, conn, target_value)

                        # Add debugging
                        if abs(new_weight) > 100:  # Still large but more reasonable
                            print(f"Large weight update: {conn.weight} -> {new_weight}")

                        if "aşırı" in diagnose.lower():
                            
                            print(f"  ! AŞIRI DEĞER DÜZELTME: {weight:.6f} -> {conn.weight:.6f}")
                        elif "zayıf" in diagnose.lower():
                            
                            print(f"  ! ZAYIF BAĞLANTI GÜÇLENDİRME: {weight:.6f} -> {conn.weight:.6f}")
                        conn.weight = normalize_weight(conn.weight + weight_update)
                        conn.bias = normalize_weight(conn.bias + bias_update)
                            
                        
                else:
                    # GİZLİ KATMANLAR için işlemler
                    if to_id == neuron.id:  # Gelen bağlantı ise
                        # Sonraki katmanın deltalarını hesapla
                        next_layer_deltas = {}
                        if not is_last_layer:
                            for next_neuron in layers[current_layer_idx + 1]:
                                if current_layer_idx == len(layers) - 2:  # Son katman öncesi
                                    next_target = target[layers[-1].index(next_neuron)] if layers[-1].index(next_neuron) < len(target) else 0
                                    next_layer_deltas[next_neuron.id] = calculate_neuron_delta(next_neuron, next_target)

                        delta = calculate_hidden_delta(neuron, next_layer_deltas)
                        weight_update = delta * from_neuron.value * learning_rateArg 

                        grad_info['grad'] = weight_update

                        # Aktivasyon kontrolü
                        if abs(from_neuron.value) < 0.0001 and neuron.activation_type == "relu":
                            print("  ! ÖLÜ ReLU TESPİT EDİLDİ")
                            from_neuron.value = 0.1  # Küçük bir değerle aktive et

                        conn.weight += weight_update  # Now modifying the Connection object's weight directly
                gradients.append(grad_info)

                # Debug çıktıları
                if debug:
                    # Renkli gösterim
                    weight_color = "\033[91m" if weight < 0 else "\033[92m"  # Kırmızı/yeşil
                    weight_str = f"{weight_color}{weight:.6f}\033[0m"
                    update_color = "\033[94m" if weight_update >=0 else "\033[95m"
                    update_str = f"{update_color}{weight_update:+.6f}\033[0m"

                    print(f"  {direction} | {connection_type} | Ağırlık: {weight_str} | Güncelleme: {update_str}")
                    print(f"    Kaynak: {from_neuron.value:.6f} | Hedef: {to_neuron.value:.6f}")

                    if to_id == neuron.id:  # Gelen bağlantılar için ekstra bilgi
                        effectiveness = weight * from_neuron.value
                        print(f"    Etki Katsayısı: {effectiveness:.6f}")
                
            # İlerleme çubuğu için hangi modda olduğumuzu belirle
            if total_epochs is not None:
                # Epoch modunda
                progress = (current_epoch / total_epochs) if total_epochs > 0 else 0
                print_epoch_progress(current_epoch, total_epochs, error)
            else:
                # Hata değeri modunda
                print_error_progress(error, target_error=perfectValueArg)
    return gradients   

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



def train_network(X_train, y_train, batch_size=256, epochs=None, intelligenceValue=None, learning_rate=0.1):
    # Kortikal kolon oluştur
    cortical_column = CorticalColumn(layers)
    
    avg_error = float('inf')
    epoch = 0
    total_samples = len(X_train)
    start_time = time.time()
    last_print_time = start_time

    # Eğitim öncesi kontrol
    if len(layers[0]) != len(X_train[0]):
        print(f"Uyarı: Giriş boyutu uyumsuz! Ağ girişi: {len(layers[0])}, Veri girişi: {len(X_train[0])}")
        return

    # Eğitim döngüsü
    while True:
        if epochs is not None and epoch >= epochs:
            break
            
        total_error = 0
        processed_samples = 0
        epoch_gradients = []  # Gradyanları topla
        
        # Batch işleme
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            batch_error = 0
            
            for X, y in zip(X_batch, y_batch):
                # İleri yayılım
                for i, val in enumerate(X[:len(layers[0])]):
                    layers[0][i].value = val
                runAI()
                
                # Hata hesapla
                output = [neuron.value for neuron in layers[-1][:len(y)]]
                error = hata_payi(y, output)
                batch_error += error
                

                # Gradyan hesapla ve kaydet
                gradients = compute_gradients(y, output)
                epoch_gradients.extend(gradients)
                # Cortical column güncellemesi
                recent_avg_error = cortical_column.monitor_network(avg_error,epoch_gradients)
                cortical_column.adapt_neurons()
                cortical_column._adapt_connections()
            
            # Batch istatistikleri
            batch_error /= len(X_batch)
            total_error += batch_error * len(X_batch)
            processed_samples += len(X_batch)
            
            # Ortalama hatayı güncelle
            avg_error = total_error / processed_samples
            
            # IntelligenceValue kontrolü (her batch sonunda)


            if intelligenceValue is not None and avg_error <= intelligenceValue:
                print(f"\nHata {intelligenceValue} değerinin altına düştü! Eğitim durduruldu.")
                break
                

            
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
                
                last_print_time = current_time
        
        # Epoch sonu işlemleri
        epoch += 1
        
        # IntelligenceValue kontrolü (her epoch sonunda)
        if intelligenceValue is not None and avg_error <= intelligenceValue:
            break
    
    # Final raporu
    total_time = time.time() - start_time
    print(f"\n=== EĞİTİM TAMAMLANDI ===")
    print(f"Toplam Süre: {total_time/60:.1f} dakika")
    print(f"Son Hata: {avg_error:.6f}")
    print(f"Toplam Epoch: {epoch}")
    print(f"Final Ağ Yapısı: {[len(layer) for layer in layers]}")
    print(f"Katman Sağlık Skorları:")
    for layer_idx, health in cortical_column.layer_health.items():
        print(f"  Katman {layer_idx}: {health['health_score']:.2f}")

"""
# Örnek veri
X_train = [[0.1, 0.9, 0.2], [0.8, 0.3, 0.5], ...]
y_train = [[1,0,0], [0,1,0], ...]

# Eğitim başlatma
train_network(X_train, y_train, epochs=100)"""

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
drawing_mode = False
drawing_canvas = None
drawing_data = np.zeros((28, 28))  # 28x28'lik çizim alanı
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
    def __init__(self, main_network_layers):
        self.monitoring_window = 5
        self.performance_history = []
        self.error_history = []
        self.gradient_history = []
        self.layer_health = {}
        self.adaptation_threshold = 0.1
        self.added_layers = set()
        self.last_changes = []
        
        # Optimize Edilmiş Parametreler
        self.max_neurons_base = 64
        self.max_neurons_multiplier = 4
        self.error_increase_threshold = 0.1
        self.strict_mode = True
        self.health_threshold = 0.45
        self.activation_target = 0.5
        self.activation_tolerance = 0.15
        self.gradient_threshold = 0.08
        
        # Katman sağlık verilerini başlat
        for i, layer in enumerate(main_network_layers):
            self.layer_health[i] = {
                'activation_mean': [],
                'gradient_mean': [],
                'health_score': 0.5,
                'added_neurons': 0,
                'dynamic_limit': self._dynamic_neuron_limit(i)
            }
        

    def _dynamic_neuron_limit(self, layer_idx):
        """Problemin boyutuna göre akıllı nöron limiti"""
        input_size = len(layers[0]) if layers else 8
        output_size = len(layers[-1]) if layers else 1
        return min(
            self.max_neurons_base * self.max_neurons_multiplier,
            max(
                self.max_neurons_base,
                input_size * 2,
                output_size * 4,
                int(np.log2(input_size + output_size) * 16)
            )
        )

    
    def monitor_network(self, epoch_error, gradients):
        current_layer_count = len(layers)
        for layer_idx in range(current_layer_count):
            if layer_idx not in self.layer_health:
                self.layer_health[layer_idx] = {
                    'activation_mean': [],
                    'gradient_mean': [],
                    'health_score': 0.4,
                    'added_neurons': 0,
                    'dynamic_limit': self._dynamic_neuron_limit(layer_idx)
                }

        # Hata artış kontrolü
        if len(self.error_history) > 0:
            last_error = self.error_history[-1]
            if (epoch_error - last_error) > self.error_increase_threshold:
                print(f"\nUYARI: Hata artışı tespit edildi! Son değişiklikler geri alınıyor...")
                self._revert_last_changes()
                return last_error

        self.error_history.append(epoch_error)
        self.gradient_history.append(gradients)

        # Katman sağlık güncellemeleri
        for layer_idx, layer in enumerate(layers):
            health_data = self.layer_health[layer_idx]
            
            # Dinamik limiti güncelle
            health_data['dynamic_limit'] = self._dynamic_neuron_limit(layer_idx)
            
            # Aktivasyon skoru
            activations = [n.value for n in layer]
            activation_mean = np.clip(np.mean(activations), -1, 1)
            activation_deviation = abs(activation_mean - self.activation_target)
            activation_score = max(0, 1 - (activation_deviation / self.activation_tolerance))
            
            # Gradyan skoru
            layer_gradients = [g['grad'] for g in gradients if g['layer'] == layer_idx]
            grad_mean = np.mean(np.abs(layer_gradients)) if layer_gradients else 0
            gradient_score = min(1, grad_mean * 10) if grad_mean > self.gradient_threshold else 0
            
            # Ağırlık skoru
            if layer_idx > 0:
                weights = [abs(c.weight) for n in layer for c in get_neuron_connections(n.id, incoming=True)]
                weight_mean = np.mean(weights) if weights else 0
                weight_score = 1 - min(1, abs(weight_mean - 0.5)/0.5)
            else:
                weight_score = 1
                
            # Sağlık skoru güncelleme
            new_health = (activation_score*0.5 + gradient_score*0.3 + weight_score*0.2)
            health_data['health_score'] = health_data['health_score']*0.8 + new_health*0.2
        



            # Bağlantı sağlık izleme ekle
        connection_health = self._analyze_connections()
        for layer_idx in connection_health:
            if layer_idx not in self.layer_health:
                continue

            layer_conn_health = [c['health_score'] for c in connection_health[layer_idx]]
            if layer_conn_health:
                conn_health_score = np.mean(layer_conn_health)
                # Katman sağlık skoruna bağlantı sağlığını da ekle (%30 ağırlık)
                self.layer_health[layer_idx]['health_score'] = (
                    self.layer_health[layer_idx]['health_score'] * 0.7 + 
                    conn_health_score * 0.3
                )

        return np.mean(self.error_history[-self.monitoring_window:])
    
    def _revert_last_changes(self):
        """Son yapılan değişiklikleri geri al (bağlantı desteği eklendi)"""
        for change in reversed(self.last_changes):
            if change['type'] == 'add_neuron':
                layer = layers[change['layer_idx']]
                if layer and layer[-1].id == change['neuron_id']:
                    layer.pop()
                    print(f"  - Nöron {change['neuron_id']} {change['layer_idx']}. katmandan kaldırıldı")

            elif change['type'] == 'remove_neuron':
                neuron = Neuron(change['value'])
                neuron.id = change['neuron_id']
                layers[change['layer_idx']].append(neuron)
                print(f"  - Nöron {change['neuron_id']} {change['layer_idx']}. katmana geri eklendi")

            elif change['type'] == 'add_layer':
                if change['layer_idx'] < len(layers):
                    removed_layer = layers.pop(change['layer_idx'])
                    print(f"  - {change['layer_idx']}. katman kaldırıldı ({len(removed_layer)} nöron)")

            elif change['type'] in ['reset_weight', 'clip_weight', 'boost_weight']:
                # Bağlantı ağırlık değişikliklerini geri al
                for layer_idx in connections:
                    if change['from'] in connections[layer_idx]:
                        for conn in connections[layer_idx][change['from']]:
                            if conn.connectedTo[1] == change['to']:
                                conn.weight = change['old_weight']
                                print(f"  - Bağlantı {change['from']}→{change['to']} ağırlığı geri alındı: {change['old_weight']:.4f}")
                                break
                            
            # Bağlantıları yeniden oluştur
            setConnections()

        # Değişiklik geçmişini temizle
        self.last_changes = []
    
    def adapt_neurons(self):
        self.last_changes = []

        # Önce bağlantı adaptasyonu yap
        conn_changes = self._adapt_connections()
        self.last_changes.extend(conn_changes)

        prune_count = self._prune_connections()
        if prune_count > 0:
            self.last_changes.append({'type': 'prune', 'count': prune_count})

        added_conns = self._add_new_connections()
        if added_conns > 0:
            self.last_changes.append({'type': 'add_conn', 'count': added_conns})

        # Sonra nöron adaptasyonu yap (orijinal kod)
        for layer_idx in list(self.layer_health.keys()):
            if layer_idx == 0 or layer_idx == len(layers)-1:
                continue

            health_data = self.layer_health[layer_idx]
            current_neurons = len(layers[layer_idx])
            max_limit = health_data['dynamic_limit']

            if (health_data['health_score'] < self.health_threshold and
                current_neurons < max_limit and
                health_data['added_neurons'] < max_limit//3):

                add_count = 1 if current_neurons < 32 else 2
                added_ids = self._add_neurons(layer_idx, add_count)
                health_data['added_neurons'] += add_count
                self.last_changes.extend([{
                    'type': 'add_neuron',
                    'layer_idx': layer_idx,
                    'neuron_id': nid
                } for nid in added_ids])

            elif (health_data['health_score'] > 0.65 and
                  health_data['added_neurons'] > 0 and
                  current_neurons > 2):

                removed = self._remove_neurons(layer_idx, 1)
                if removed:
                    health_data['added_neurons'] -= 1
                    self.last_changes.append({
                        'type': 'remove_neuron',
                        'layer_idx': layer_idx,
                        'neuron_id': removed.id,
                        'value': removed.value
                    })
    def _add_neurons(self, layer_idx, count):
        """Akıllı nöron ekleme mekanizması"""
        if layer_idx <= 0 or layer_idx >= len(layers)-1:
            return []

        new_neurons = []
        for _ in range(count):
            # Mevcut aktivasyonlara göre akıllı başlangıç değeri
            layer_activations = [n.value for n in layers[layer_idx]]
            base_value = np.clip(np.mean(layer_activations), -0.5, 0.5)
            new_neuron = Neuron(random.uniform(base_value-0.1, base_value+0.1))
            new_neurons.append(new_neuron)
        
        layers[layer_idx].extend(new_neurons)
        setConnections()
        return [n.id for n in new_neurons]

    def _remove_neurons(self, layer_idx, count):
        """En az katkı sağlayan nöronları çıkar"""
        if len(layers[layer_idx]) <= 2:
            return None
            
        # Aktivasyon ve gradyanlara göre önem sıralaması
        neuron_scores = []
        for neuron in layers[layer_idx]:
            incoming = [abs(c.weight) for c in get_neuron_connections(neuron.id, incoming=True)]
            outgoing = [abs(c.weight) for c in get_neuron_connections(neuron.id, outgoing=True)]
            score = (np.mean(incoming) if incoming else 0) + (np.mean(outgoing) if outgoing else 0)
            neuron_scores.append((score, neuron))
            
        # En düşük skorlu nöronları seç
        neuron_scores.sort(key=lambda x: x[0])
        removed = []
        for _ in range(min(count, len(neuron_scores))):
            if neuron_scores[0][0] < 0.1:  # Yalnızca çok düşük katkılıları çıkar
                removed.append(layers[layer_idx].pop(layers[layer_idx].index(neuron_scores[0][1])))
            del neuron_scores[0]
            
        setConnections()
        return removed[0] if removed else None

    def _add_layer(self, layer_idx):
        """Add a new layer after the specified layer index."""
        # Ensure we're adding between existing layers (not after output or before input)
        if layer_idx <= 0 or layer_idx >= len(layers):
            print(f"Warning: Invalid layer addition position {layer_idx}")
            return

        # Add new layer with size based on adjacent layers
        prev_size = len(layers[layer_idx-1])
        next_size = len(layers[layer_idx]) if layer_idx < len(layers) else prev_size
        new_size = max(2, min(prev_size, next_size))  # At least 2 neurons

        new_layer = [Neuron(random.uniform(-0.1, 0.1)) for _ in range(new_size)]
        layers.insert(layer_idx, new_layer)

        # Update layer health tracking
        self.layer_health[layer_idx] = {
            'activation_mean': [],
            'gradient_mean': [],
            'health_score': 0.5,  # Start with medium health
            'added_neurons': 0
        }

        # Shift existing layer health data for layers after the new one
        for i in range(len(layers)-1, layer_idx, -1):
            if i in self.layer_health:
                self.layer_health[i+1] = self.layer_health[i]

        setConnections()
        print(f"Added new layer at position {layer_idx} with {new_size} neurons")
    def _analyze_connections(self):
        """Bağlantı sağlık analizi yapar"""
        connection_health = {}
        for layer_idx in connections:
            layer_health = []
            for src_id in connections[layer_idx]:
                for conn in connections[layer_idx][src_id]:
                    to_neuron = get_neuron_by_id(conn.connectedTo[1])
                    from_neuron = get_neuron_by_id(src_id)
                    
                    # Bağlantı sağlık metriği
                    weight_health = 1 - min(abs(conn.weight)/5, 1)  # 0-1 arası
                    activation_health = 1 - abs(from_neuron.value - 0.5)  # 0.5'e yakın olmalı
                    gradient_health = min(abs(conn.weight * from_neuron.value)*10, 1)
                    
                    health_score = (weight_health + activation_health + gradient_health) / 3
                    layer_health.append({
                        'connection': conn,
                        'health_score': health_score,
                        'from': src_id,
                        'to': conn.connectedTo[1],
                        'layer': layer_idx
                    })
            connection_health[layer_idx] = layer_health
        return connection_health
    
    def _adapt_connections(self):
        """Bağlantıları akıllıca adapte eder"""
        connection_health = self._analyze_connections()
        changes = []
        
        for layer_idx in connection_health:
            for conn_data in connection_health[layer_idx]:
                conn = conn_data['connection']
                health = conn_data['health_score']
                
                # Sağlıksız bağlantıları düzelt
                if health < self.health_threshold:
                    # 1. Ağırlık sıfıra yakınsa yeniden başlat
                    if abs(conn.weight) < 0.01:
                        new_weight = random.uniform(-1/np.sqrt(len(layers[layer_idx])), 
                                         1/np.sqrt(len(layers[layer_idx])))
                        changes.append({
                            'type': 'reset_weight',
                            'from': conn_data['from'],
                            'to': conn_data['to'],
                            'old_weight': conn.weight,
                            'new_weight': new_weight
                        })
                        conn.weight = new_weight
                    
                    # 2. Aşırı büyük ağırlıkları kırp
                    elif abs(conn.weight) > 2.0:
                        new_weight = np.clip(conn.weight, -2.0, 2.0)
                        changes.append({
                            'type': 'clip_weight',
                            'from': conn_data['from'],
                            'to': conn_data['to'],
                            'old_weight': conn.weight,
                            'new_weight': new_weight
                        })
                        conn.weight = new_weight
                    
                    # 3. Ölü bağlantıları canlandır
                    elif abs(conn.weight * get_neuron_by_id(conn_data['from']).value) < 0.001:
                        boost = random.uniform(0.5, 1.5)
                        new_weight = conn.weight * boost
                        changes.append({
                            'type': 'boost_weight',
                            'from': conn_data['from'],
                            'to': conn_data['to'],
                            'old_weight': conn.weight,
                            'new_weight': new_weight
                        })
                        conn.weight = new_weight
        
        return changes
    
    def _prune_connections(self):
        """Gereksiz bağlantıları budar"""
        to_prune = []
        connection_health = self._analyze_connections()
        
        for layer_idx in connection_health:
            for conn_data in connection_health[layer_idx]:
                # Çok zayıf ve sağlıksız bağlantıları işaretle
                if (conn_data['health_score'] < 0.2 and 
                    abs(conn_data['connection'].weight) < 0.05):
                    to_prune.append(conn_data)
        
        # En kötü %10'u sil
        to_prune.sort(key=lambda x: x['health_score'])
        prune_count = max(1, int(len(to_prune) * 0.1))
        
        for conn_data in to_prune[:prune_count]:
            layer_idx = conn_data['layer']
            src_id = conn_data['from']
            conn_list = connections[layer_idx][src_id]
            
            # Bağlantıyı bul ve sil
            for i, conn in enumerate(conn_list):
                if conn.connectedTo[1] == conn_data['to']:
                    conn_list.pop(i)
                    break
                
        return len(to_prune[:prune_count])
    
    def _add_new_connections(self):
        """Yeni bağlantılar ekler"""
        added = 0
        for layer_idx in range(len(layers)-1):
            # Katmanlar arasında eksik bağlantıları bul
            existing_pairs = set()
            for src_id in connections[layer_idx]:
                for conn in connections[layer_idx][src_id]:
                    existing_pairs.add((src_id, conn.connectedTo[1]))
            
            # Rastgele yeni bağlantılar ekle (en fazla 2)
            possible_pairs = []
            for src in layers[layer_idx]:
                for dst in layers[layer_idx+1]:
                    if (src.id, dst.id) not in existing_pairs:
                        possible_pairs.append((src.id, dst.id))
            
            if possible_pairs:
                new_pairs = random.sample(possible_pairs, min(2, len(possible_pairs)))
                for src_id, dst_id in new_pairs:
                    if src_id not in connections[layer_idx]:
                        connections[layer_idx][src_id] = []
                    
                    new_weight = random.uniform(-1/np.sqrt(len(layers[layer_idx])), 
                                     1/np.sqrt(len(layers[layer_idx])))
                    connections[layer_idx][src_id].append(
                        Connection(fromTo=[src_id, dst_id], weight=new_weight)
                    )
                    added += 1
        return added



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
            setLayers(64,32,10)
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
            # Parametreleri ayrıştır
            params = cmd[len("train_custom("):-1].split(";")

            if len(params) < 1 or not params[0].strip():
                raise ValueError("Dosya yolu gereklidir")

            file_path = params[0].strip()

            # Varsayılan değerler
            network_structure = [2, 4, 1]  # Varsayılan ağ yapısı: 2-4-1
            epochs = 10
            batch_size = 128
            learning_rate = 0.1
            intelligence = None

            # Parametreleri parse et
            param_index = 1
            if len(params) > param_index and params[param_index].strip():
                # Ağ yapısı parametresi (örn: "2,4,1")
                if "[" in params[param_index] or "(" in params[param_index]:
                    # Köşeli veya normal parantezli girişler için
                    network_structure = eval(params[param_index])
                else:
                    # Virgülle ayrılmış değerler için
                    network_structure = list(map(int, params[param_index].split(",")))
                param_index += 1

            if len(params) > param_index and params[param_index].strip(): epochs = float(params[param_index])
            if epochs <1.0:
                intelligence = epochs
                epochs=None
            else:
                epochs = int(epochs)
            param_index += 1
            if len(params) > param_index and params[param_index].strip(): batch_size = int(params[param_index])
            param_index += 1
            if len(params) > param_index and params[param_index].strip(): learning_rate = float(params[param_index])
            param_index += 1
            if len(params) > param_index and params[param_index].strip(): intelligence = float(params[param_index])

            # Ağ yapısını oluştur
            setLayers(*network_structure)

            print(f"\nÖzel Veri Seti Eğitim Parametreleri:")
            print(f"- Dosya: {file_path}")
            print(f"- Ağ Yapısı: {network_structure}")
            print(f"- Epochs: {epochs if epochs is not None else 'Kapalı'}")
            print(f"- Batch Size: {batch_size}")
            print(f"- Learning Rate: {learning_rate}")
            print(f"- Intelligence Threshold: {intelligence if intelligence is not None else 'Kapalı'}")

            # Verileri yükle ve eğit
            X, y = modeltrainingprogram.read_csv_file(file_path)
            train_network(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                intelligenceValue=intelligence
            )

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
    elif cmd.startswith("add_neuron("):
        try:
            args = cmd[11:-1].split(",")
            layer_idx = int(args[0])
            new_neuron_value = float(args[1])
            
            new_neuron = Neuron(default_value=new_neuron_value)
            add_neuron(layers, connections, layer_idx, new_neuron)
            print(f"{layer_idx}. katmana nöron eklendi. Yeni nöron ID: {new_neuron.id}")
            
        except Exception as error:
            print("Hatalı giriş! Örnek format: add_neuron(layerId,newNeuronValue)")
            traceback.print_exc()
    
    elif cmd.startswith("removeLayer("):
        try:
            layer_idx = int(cmd[12:-1])
            removeLayer(layers, connections, layer_idx)
        except Exception as error:
            print("Hatalı giriş! Örnek format: removeLayer(1)")
            traceback.print_exc()

    elif cmd.startswith("removeNeuron("):
        try:
            neuron_id = int(cmd[13:-1])
            removeNeuron(layers, connections, neuron_id)
        except Exception as error:
            print("Hatalı giriş! Örnek format: removeNeuron(5)")
            traceback.print_exc()

    elif cmd.startswith("addLayer("):
        try:
            args_str = cmd[9:-1]
            parts = args_str.split(";", 1)
            layer_idx = int(parts[0].strip())

            neurons = []
            if len(parts) > 1 and parts[1].strip():
                neuron_strs = parts[1].split(",")
                for neuron_str in neuron_strs:
                    neuron_str = neuron_str.strip()
                    if neuron_str.startswith("Neuron("):
                        value_str = neuron_str[7:-1]
                        value = float(value_str)
                        neurons.append(Neuron(value))

            if not neurons:
                prev_size = len(layers[layer_idx-1]) if layer_idx > 0 else len(layers[0])
                neurons = [Neuron(0.0) for _ in range(prev_size)]

            addLayer(layers, connections, layer_idx, neurons)
            print(f"Katman {layer_idx} başarıyla eklendi. {len(neurons)} nöron içeriyor.")

        except Exception as error:
            print("Hatalı giriş! Örnek formatlar:")
            print("addLayer(1)  # Varsayılan nöronlarla katman ekler")
            print("addLayer(1;Neuron(1),Neuron(0.2),Neuron(-0.5))  # Özel nöronlarla katman ekler")
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