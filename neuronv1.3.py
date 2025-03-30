import csv
import time
import traceback
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
        for layer_idx in range(len(layers) - 1):
            for prev_neuron in layers[layer_idx]:
                for conn in connections[layer_idx].get(prev_neuron.id, []):
                    if conn.connectedTo[1] == self.id:
                        weighted_sum += prev_neuron.value * conn.weight
        self.weightedSum = weighted_sum
        self.value = self.activation(weighted_sum)  # Aktivasyon fonksiyonunu uygula
        self.output = self.value  # Çıkışı güncelle
        return self.value





class Connection:
    def __init__(self, weight=0, fromTo=[0, 0]):
        self.weight = weight
        self.connectedTo = fromTo
    
    def update_weight(self, learning_rate, delta):
        self.weight = learning_rate * delta



visualizeNetwork =False


# Ağ oluşturma
randomMinWeight = -5.0
randomMaxWeight = 5.0

activation_types = ['sigmoid', 'tanh', 'relu']
defaultNeuronActivationType='sigmoid'


# Ağ Yapısı
input_size = 28*28  # 28x28 piksel girişi
hidden_size = 128  # Gizli katmanda 128 nöron
output_size = 10  # 0-9 için 10 çıktı nöronu
layers = [
    [Neuron(1) for i in range(input_size)],  # Giriş katmanı (784 nöron)
    [Neuron(1) for i in range(hidden_size)],  # Gizli katman (128 nöron)
    [Neuron(1) for i in range(output_size)]   # Çıkış katmanı (10 nöron)
]

# Bağlantıları oluşturma
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




# Global değişkenler
global win, plot, scatter, lines, app
win = None
plot = None
scatter = None
lines = []
app = None

cmd = "refresh()" #program başlar başlamaz çalışacak ilk komut




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



def get_neuron_by_id(neuron_id, layers=layers):
    for layer in layers:
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








previous_updates = {}
debug = True  # Global debug değişkeni

def TrainFor(inputValues, targetValues, connections, learning_rate=0.1, boost_factor=2.0, momentum=0.9, max_grad_norm=1.0):
    global previous_updates, debug
    
    try:
        # 1. Giriş verilerini kontrol et ve normalize et
        if len(inputValues) == 0 or np.all(np.isnan(inputValues)):
            print("Hata: Geçersiz giriş verisi!")
            return
            
        inputValues = np.array(inputValues, dtype=np.float32)
        input_min = np.min(inputValues)
        input_max = np.max(inputValues)
        
        # Sıfır bölme hatasını önle
        if input_max - input_min < 1e-8:
            inputValues = np.zeros_like(inputValues) + 0.5
        else:
            inputValues = (inputValues - input_min) / (input_max - input_min) * 0.8 + 0.1
        
        # 2. İleri yayılım
        for i, value in enumerate(inputValues[:len(layers[0])]):
            layers[0][i].value = value
        
        runAI()

        # 3. Aktivasyon değerlerini kontrol et
        for layer in layers:
            for neuron in layer:
                if np.isnan(neuron.value) or np.isinf(neuron.value):
                    neuron.value = 0.0
                neuron.value = np.clip(neuron.value, -100, 100)

        # 4. Hata ve delta hesaplama
        output_layer = layers[-1]
        errors = []
        deltas = {}
        
        # Çıkış katmanı deltaları
        for i, neuron in enumerate(output_layer[:len(targetValues)]):
            error = targetValues[i] - neuron.value
            errors.append(error)
            
            try:
                deriv = neuron.activation_derivative()
                if np.isnan(deriv) or deriv < 1e-8:
                    deriv = 0.1
                    if debug:
                        print(f"DEBUG: Düşük türev düzeltildi (Nöron {neuron.id}): {deriv:.6f}")
                
                delta = error * deriv
                deltas[neuron.id] = delta

            except Exception as e:
                print(f"Aktivasyon türevi hatası (Nöron {neuron.id}): {str(e)}")
                deltas[neuron.id] = 0.01

        # Gizli katman deltaları
        for layer_idx in reversed(range(len(layers)-1)):
            for neuron in layers[layer_idx]:
                delta_sum = 0
                
                if neuron.id in connections[layer_idx]:
                    for conn in connections[layer_idx][neuron.id]:
                        to_id = conn.connectedTo[1]
                        if to_id in deltas:
                            delta_sum += conn.weight * deltas[to_id]
                
                try:
                    deriv = neuron.activation_derivative()
                    if np.isnan(deriv) or deriv < 1e-8:
                        deriv = 0.1
                    
                    delta = delta_sum * deriv
                    delta = np.clip(delta, -1.0, 1.0)
                    deltas[neuron.id] = delta
                    
                except Exception as e:
                    print(f"Gizli katman türev hatası (Nöron {neuron.id}): {str(e)}")
                    deltas[neuron.id] = 0.0

        # 5. Gradyan hesaplama ve ağırlık güncelleme
        all_gradients = []
        grad_debug_info = []
        updated_connections = 0
        
        for layer_idx in range(len(layers)-1):
            if layer_idx in connections:
                for src_id, conn_list in connections[layer_idx].items():
                    src_neuron = get_neuron_by_id(src_id)
                    if src_neuron:
                        for conn in conn_list:
                            to_id = conn.connectedTo[1]
                            if to_id in deltas:
                                try:
                                    grad = deltas[to_id] * src_neuron.value
                                    
                                    if np.isnan(grad) or np.isinf(grad):
                                        grad = 0.01
                                    elif abs(grad) < 1e-8:
                                        grad = np.sign(grad) * 1e-4
                                    
                                    grad = np.clip(grad, -max_grad_norm, max_grad_norm)
                                    all_gradients.append(abs(grad))
                                    
                                    grad_debug_info.append({
                                        'from': src_id,
                                        'to': to_id,
                                        'grad': grad,
                                        'delta': deltas[to_id],
                                        'src_val': src_neuron.value,
                                        'weight': conn.weight,
                                        'layer': layer_idx
                                    })
                                    
                                    update_id = id(conn)
                                    prev_update = previous_updates.get(update_id, 0)
                                    update = learning_rate * grad + momentum * prev_update
                                    conn.weight += update
                                    previous_updates[update_id] = update
                                    updated_connections += 1
                                    
                                except Exception as e:
                                    print(f"Gradyan hesaplama hatası ({src_id}->{to_id}): {str(e)}")

        # 6. Sonuçları raporla
        output_values = [neuron.value for neuron in layers[-1][:len(targetValues)]]
        current_error = hata_payi(targetValues[:len(output_values)], output_values)
        
        print("\n=== EĞİTİM ÖZETİ ===")
        print(f"Hata: {current_error:.6f}")
        print(f"Güncellenen bağlantı sayısı: {updated_connections}")
        
        if debug:
            print(f"Gradyan bilgisi toplandı: {len(grad_debug_info)} bağlantı")
            
            if grad_debug_info:
                grads = [info['grad'] for info in grad_debug_info]
                print(f"Gradyan istatistikleri - Min: {np.min(grads):.6f}, Max: {np.max(grads):.6f}, Ort: {np.mean(grads):.6f}")
                
                weak_connections = sorted(grad_debug_info, key=lambda x: abs(x['grad']))[:5]
                print("\nEn zayıf 5 bağlantı:")
                for conn in weak_connections:
                    print(f"{conn['from']}->{conn['to']} (Katman {conn['layer']}): "
                          f"Grad={conn['grad']:.6f}, Ağırlık={conn['weight']:.6f}")
            else:
                print("UYARI: Hiç gradyan bilgisi toplanamadı!")
                
                # Gradyanların neden hesaplanamadığını araştır
                print("\nDEBUG: Potansiyel nedenler:")
                print(f"- Delta değerleri: {len(deltas)}")
                print(f"- Bağlantı sayısı: {sum(len(c) for c in connections.values())}")
                
                # Deltaları kontrol et
                if not deltas:
                    print("- Hiç delta hesaplanamadı (aktivasyon problemleri olabilir)")
                else:
                    print(f"- Delta değer aralığı: Min {min(deltas.values()):.6f}, Max {max(deltas.values()):.6f}")
                
                # Nöron değerlerini kontrol et
                print("\nNöron değer özeti:")
                for i, layer in enumerate(layers):
                    values = [n.value for n in layer]
                    print(f"Katman {i}: Min {min(values):.6f}, Max {max(values):.6f}, Ort {np.mean(values):.6f}")
        
    except Exception as e:
        print(f"Eğitim hatası: {str(e)}")
        traceback.print_exc()





def clearGUI():
    global win
    if win is not None:
        win.close()
        win = None



class DynamicNetworkManager:
    def __init__(self, layers, connections):
        self.layers = layers
        self.connections = connections
        self.error_history = []
        self.activation_history = {i: [] for i in range(len(layers))}
        self.learning_rate = 0.1
        self.scaling_factor = 0.5
        self.current_target = None
        self.debug = True  # Debug modu için class seviyesinde flag

    def adapt_network(self, current_error, current_target=None,perfectValueArg=0.3):
        """Ağın performansına göre tüm adaptasyon operasyonlarını yönetir"""
        if self.debug:
            print(f"ADAPTASYON | Güncel hata: {current_error:.6f}")
            
        self.error_history.append(current_error)
        if current_target is not None:
            self.current_target = current_target
            if self.debug:
                print(f"ADAPTASYON | Hedefler güncellendi: {current_target}")
                
        self._modify_architecture(current_error,perfectValue=perfectValueArg)
        
    def _modify_architecture(self, avg_error, perfectValue=0.3):
        """Akıllı mimari adaptasyonu"""
        if self.debug:
            print(f"OPTİMİZE | Ortalama hata: {avg_error:.6f}, Hedef eşik: {perfectValue}")
            
        # 1. Nöron/Katman optimizasyonu
        if self.current_target is not None:
            if self.debug:
                print("OPTİMİZE | Bağlantı optimizasyonu başlatılıyor...")
            self._optimize_connections()
        
        # 2. Stratejik eklemeler
        if avg_error > perfectValue:
            if self.debug:
                print("OPTİMİZE | Hata yüksek, stratejik eklemeler yapılıyor...")
            self._add_neuron_strategically()
            self._add_layer_strategically()
        
        # 3. Temizleme operasyonları
        if self.debug:
            print("OPTİMİZE | Ağ temizleme işlemleri başlatılıyor...")
        self._prune_network()

    def _calculate_required_change(self, neuron, target):
        """Nöronun hedef değere ulaşması için gereken değişim miktarını hesaplar"""
        current = neuron.value
        delta = target - current
        change = delta * self.scaling_factor * neuron.activation_derivative()
        
        if self.debug and abs(change) > 1.0:
            print(f"OPTİMİZE | Büyük değişim gerekiyor (Nöron {neuron.id}): {change:.6f}")
            
        return change

    def _optimize_connections(self):
        """Hedef-çıktı farkına göre bağlantıları optimize eder"""
        if self.current_target is None:
            if self.debug:
                print("OPTİMİZE | Hedef yok, optimizasyon atlandı")
            return
            
        output_layer = self.layers[-1]
        updated_connections = 0
        
        for neuron_idx, output_neuron in enumerate(output_layer):
            if neuron_idx >= len(self.current_target):
                continue
                
            target = self.current_target[neuron_idx]
            delta = self._calculate_required_change(output_neuron, target)
            
            for layer_idx in reversed(range(len(self.layers)-1)):
                for src_id, connections in self.connections[layer_idx].items():
                    for conn in connections:
                        if conn.connectedTo[1] == output_neuron.id:
                            src_neuron = self._get_neuron_by_id(src_id)
                            adjustment = delta * src_neuron.value * self.learning_rate
                            
                            if abs(adjustment) > 0.5:
                                conn.weight += adjustment * 2.0
                                if self.debug:
                                    print(f"OPTİMİZE | Büyük güncelleme ({src_id}->{output_neuron.id}): {adjustment*2.0:.6f}")
                            elif abs(adjustment) < 0.01:
                                conn.weight += adjustment * 0.1
                            else:
                                conn.weight += adjustment
                            
                            updated_connections += 1
        
        if self.debug:
            print(f"OPTİMİZE | Toplam {updated_connections} bağlantı güncellendi")

    def _add_neuron_strategically(self, similarity_threshold=0.15):
        """Akıllı nöron ekleme mekanizması"""
        for layer_idx in range(1, len(self.layers)-1):
            next_layer = self.layers[layer_idx+1]
            values = [n.value for n in next_layer]
            value_range = max(values) - min(values)
            
            if self.debug:
                print(f"OPTİMİZE | Katman {layer_idx+1} değer aralığı: {value_range:.6f}")
            
            if value_range < similarity_threshold:
                new_neuron = Neuron()
                self.layers[layer_idx].append(new_neuron)
                if self.debug:
                    print(f"OPTİMİZE | Yeni nöron eklendi - Katman: {layer_idx}, ID: {new_neuron.id}, Sebep: Düşük çeşitlilik ({value_range:.6f} < {similarity_threshold})")
                
                self._reconnect_layer(layer_idx)
                return

    def _add_layer_strategically(self):
        """Akıllı katman ekleme mekanizması"""
        for layer_idx in range(1, len(self.layers)-1):
            current_layer = self.layers[layer_idx]
            activations = [n.value for n in current_layer]
            activation_std = np.std(activations)
            
            if self.debug:
                print(f"OPTİMİZE | Katman {layer_idx} aktivasyon std: {activation_std:.6f}")
            
            if activation_std < 0.1:
                new_size = max(2, len(current_layer)//2)
                new_layer = [Neuron() for _ in range(new_size)]
                self.layers.insert(layer_idx+1, new_layer)
                if self.debug:
                    print(f"OPTİMİZE | Yeni katman eklendi - Pozisyon: {layer_idx+1}, Boyut: {new_size}, Sebep: Düşük aktivasyon çeşitliliği ({activation_std:.6f})")
                
                self._reconnect_layer(layer_idx)
                self._reconnect_layer(layer_idx+1)
                return

    def _prune_network(self, threshold=0.05):
        """Performansı düşük elemanları temizler"""
        # 1. Zayıf nöronları bul
        inactive_neurons = self._find_inactive_neurons(threshold)
        if self.debug:
            print(f"OPTİMİZE | Potansiyel silinecek nöronlar: {len(inactive_neurons)} (Eşik: {threshold})")
        
        for neuron_id in inactive_neurons:
            self.removeNeuron(neuron_id)
            if self.debug:
                print(f"OPTİMİZE | Nöron silindi: {neuron_id}")
            
        # 2. Etkisiz katmanları temizle
        weakest_layer = self._find_weakest_layer()
        if len(self.layers) > 3 and weakest_layer not in [0, len(self.layers)-1]:
            if self.debug:
                layer_error = np.mean([abs(n.value - np.mean(n.value)) for n in self.layers[weakest_layer]])
                print(f"OPTİMİZE | En zayıf katman: {weakest_layer}, Hata: {layer_error:.6f}")
            self.removeLayer(weakest_layer)

    def _find_inactive_neurons(self, threshold=0.05):
        """Düşük aktiviteli nöronları tespit eder"""
        inactive = []
        for layer_idx, layer in enumerate(self.layers[1:-1]):
            for neuron in layer:
                avg_activation = np.mean(self.activation_history[layer_idx])
                if avg_activation < threshold:
                    inactive.append(neuron.id)
                    if self.debug:
                        print(f"OPTİMİZE | Düşük aktivasyon - Nöron: {neuron.id}, Ortalama: {avg_activation:.6f}, Eşik: {threshold}")
        return inactive

    def _find_weakest_layer(self):
        """En düşük performanslı katmanı bulur"""
        layer_errors = []
        for layer_idx in range(1, len(self.layers)-1):
            layer_error = np.mean([abs(n.value - np.mean(n.value)) for n in self.layers[layer_idx]])
            layer_errors.append(layer_error)
        return np.argmin(layer_errors) + 1

    def _reconnect_layer(self, layer_idx):
        """Katman bağlantılarını yeniden oluşturur"""
        # Önceki bağlantıları temizle
        if layer_idx-1 in self.connections:
            del self.connections[layer_idx-1]
        if layer_idx in self.connections:
            del self.connections[layer_idx]
            
        # Yeni bağlantıları rastgele oluştur
        if layer_idx > 0:
            self.connections[layer_idx-1] = {}
            for prev_neuron in self.layers[layer_idx-1]:
                for curr_neuron in self.layers[layer_idx]:
                    conn = Connection(fromTo=[prev_neuron.id, curr_neuron.id],
                                    weight=random.uniform(randomMinWeight, randomMaxWeight))
                    if prev_neuron.id not in self.connections[layer_idx-1]:
                        self.connections[layer_idx-1][prev_neuron.id] = []
                    self.connections[layer_idx-1][prev_neuron.id].append(conn)
                    
        if layer_idx < len(self.layers)-1:
            self.connections[layer_idx] = {}
            for curr_neuron in self.layers[layer_idx]:
                for next_neuron in self.layers[layer_idx+1]:
                    conn = Connection(fromTo=[curr_neuron.id, next_neuron.id],
                                    weight=random.uniform(randomMinWeight, randomMaxWeight))
                    if curr_neuron.id not in self.connections[layer_idx]:
                        self.connections[layer_idx][curr_neuron.id] = []
                    self.connections[layer_idx][curr_neuron.id].append(conn)

    # Yardımcı fonksiyonlar
    def _get_neuron_by_id(self, neuron_id):
        for layer in self.layers:
            for neuron in layer:
                if neuron.id == neuron_id:
                    return neuron
        return None

    def removeNeuron(self, neuron_id):
        # Var olan removeNeuron fonksiyonuyla entegrasyon
        removeNeuron(self.layers, self.connections, neuron_id)

    def removeLayer(self, layer_idx):
        # Var olan removeLayer fonksiyonuyla entegrasyon
        removeLayer(self.layers, self.connections, layer_idx)
        



def train_network(X_train, y_train, batch_size=256, epochs=None, intelligenceValue=None, learning_rate=0.1):
    dynamic_manager = DynamicNetworkManager(layers, connections)
    avg_error = float('inf')
    epoch = 0
    
    total_samples = len(X_train)
    start_time = time.time()
    last_print_time = start_time
    
    while True:
        if epochs is not None and epoch >= epochs:
            break
        if intelligenceValue is not None and avg_error <= intelligenceValue:
            print(f"\nHata {intelligenceValue} değerinin altına düştü! Eğitim durduruldu.")
            break
            
        total_error = 0
        processed_samples = 0
        
        # Batch'ler halinde işleme
        for batch_start in range(0, total_samples, batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            
            # Batch için ortalama hata
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
                
                # Geri yayılım
                TrainFor(X, y, connections, learning_rate=learning_rate)
                # Ağ yapısını adapte et
                if intelligenceValue is not None:
                    dynamic_manager.adapt_network(error, current_target=y, perfectValueArg=intelligenceValue)
                else:
                    dynamic_manager.adapt_network(error, current_target=y)
            
            # Batch istatistiklerini güncelle
            batch_error /= len(X_batch)
            total_error += batch_error * len(X_batch)
            processed_samples += len(X_batch)
            
            # Her 10 saniyede bir veya her epoch başında ilerlemeyi göster
            current_time = time.time()
            if current_time - last_print_time > 10 or batch_end >= total_samples:
                elapsed_time = current_time - start_time
                samples_per_sec = processed_samples / elapsed_time if elapsed_time > 0 else 0
                remaining_samples = total_samples - processed_samples
                remaining_time = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                
                print(f"\nEpoch {epoch+1}/{epochs if epochs else '∞'} - İlerleme: {processed_samples}/{total_samples} ({100*processed_samples/total_samples:.1f}%)")
                print(f"Ortalama Hata: {total_error/processed_samples:.4f}")
                print(f"Geçen Süre: {elapsed_time/60:.1f} dak - Tahmini Kalan Süre: {remaining_time/60:.1f} dak")
                print(f"Örnek/Saniye: {samples_per_sec:.1f}")
                
                last_print_time = current_time
        
        avg_error = total_error / total_samples
        epoch += 1
        
        # Her epoch sonunda kısa özet
        print(f"Epoch {epoch} tamamlandı - Ortalama Hata: {avg_error:.4f}")
        
        if epoch % 10 == 0:
            visualize_network(layers, connections, refresh=True)
    
    total_time = time.time() - start_time
    print(f"\nEğitim tamamlandı! Toplam Süre: {total_time/60:.1f} dakika")
    print(f"Son Hata: {avg_error:.4f}")
    print(f"Toplam Epoch: {epoch}")
    print(f"Ortalama Epoch Süresi: {total_time/max(1, epoch):.1f} saniye")

"""
# Örnek veri
X_train = [[0.1, 0.9, 0.2], [0.8, 0.3, 0.5], ...]
y_train = [[1,0,0], [0,1,0], ...]

# Eğitim başlatma
train_network(X_train, y_train, epochs=100)"""

def getOutput():
    outputValues=[]
    for neuronOnLastLayer in layers[-1]:
        outputValues.append(str(neuronOnLastLayer.value) + " " + str(neuronOnLastLayer.weightedSum))
    return outputValues







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
        brush_size = 1  # Digits için küçük fırça
    
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
    
    # Ağ yapısını MNIST için ayarla
    layers = [
        [Neuron(activation_type='sigmoid') for _ in range(784)],  # 28x28 = 784
        [Neuron(activation_type='relu') for _ in range(256)],     # Gizli katman
        [Neuron(activation_type='sigmoid') for _ in range(10)]    # Çıkış (0-9)
    ]
    
    # Bağlantıları yeniden oluştur
    connections = {layer_idx: {} for layer_idx in range(len(layers)-1)}
    for layer_idx in range(len(layers)-1):
        for neuron in layers[layer_idx]:
            for next_neuron in layers[layer_idx+1]:
                conn = Connection(
                    fromTo=[neuron.id, next_neuron.id],
                    weight=random.uniform(-1/np.sqrt(len(layers[layer_idx])), 
                    1/np.sqrt(len(layers[layer_idx]))))
                if neuron.id not in connections[layer_idx]:
                    connections[layer_idx][neuron.id] = []
                connections[layer_idx][neuron.id].append(conn)
    
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
    
    # Ağ yapısını Digits için ayarla (8x8=64)
    layers = [
        [Neuron(activation_type='sigmoid') for _ in range(64)],
        [Neuron(activation_type='relu') for _ in range(32)],
        [Neuron(activation_type='sigmoid') for _ in range(10)]
    ]
    
    # Bağlantıları yeniden oluştur
    connections = {layer_idx: {} for layer_idx in range(len(layers)-1)}
    for layer_idx in range(len(layers)-1):
        for neuron in layers[layer_idx]:
            for next_neuron in layers[layer_idx+1]:
                conn = Connection(
                    fromTo=[neuron.id, next_neuron.id],
                    weight=random.uniform(-1/np.sqrt(len(layers[layer_idx])), 
                    1/np.sqrt(len(layers[layer_idx]))))
                if neuron.id not in connections[layer_idx]:
                    connections[layer_idx][neuron.id] = []
                connections[layer_idx][neuron.id].append(conn)
    
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
            # Parametreleri ayrıştır (dosya yolu zorunlu)
            params = cmd[len("train_custom("):-1].split(";")
            
            if len(params) < 1 or not params[0].strip():
                raise ValueError("Dosya yolu gereklidir")
            
            file_path = params[0].strip()
            
            # Varsayılan değerler
            epochs = 10
            batch_size = 128
            learning_rate = 0.1
            test_size = 0.2
            intelligence = None
            
            # Kullanıcı parametrelerini al
            if len(params) > 1 and params[1].strip(): epochs = int(params[1])
            if len(params) > 2 and params[2].strip(): batch_size = int(params[2])
            if len(params) > 3 and params[3].strip(): learning_rate = float(params[3])
            if len(params) > 4 and params[4].strip(): test_size = float(params[4])
            if len(params) > 5 and params[5].strip(): intelligence = float(params[5])
            
            print(f"\nÖzel Veri Seti Eğitim Parametreleri:")
            print(f"- Dosya: {file_path}")
            print(f"- Epochs: {epochs}")
            print(f"- Batch Size: {batch_size}")
            print(f"- Learning Rate: {learning_rate}")
            print(f"- Test Size: {test_size}")
            print(f"- Intelligence Threshold: {intelligence if intelligence is not None else 'Kapalı'}")
            
            X, y = modeltrainingprogram.read_csv_file(file_path)
            train_network(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                test_size=test_size,
                intelligenceValue=intelligence
            )
            
        except Exception as e:
            print("Hatalı komut formatı! Örnek kullanımlar:")
            print("train_custom('veri.csv')  # Sadece dosya yolu")
            print("train_custom('veri.csv';5)  # Dosya yolu ve epoch")
            print("train_custom('veri.csv';10;64;0.05;0.1;0.01)  # Tüm parametreler")
            print("Parametre sırası: file_path;epochs;batch_size;learning_rate;test_size;intelligence")
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