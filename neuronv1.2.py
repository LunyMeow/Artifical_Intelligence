import csv
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
            return self.output * (1 - self.output)  # f'(x) = f(x)(1 - f(x))
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
            conn = Connection(fromTo=[neuron.id, next_neuron.id], weight=random.uniform(randomMinWeight, randomMaxWeight))
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


def TrainFor(inputValues, targetValues, connections, learning_rate=0.1, boost_factor=2.0, momentum=0.9, max_grad_norm=1.0):
    global previous_updates
    
    try:
        # 1. Girişleri normalize et ve çeşitlendir
        inputValues = np.array(inputValues)
        inputValues = (inputValues - np.min(inputValues)) / (np.max(inputValues) - np.min(inputValues) + 1e-8)
        inputValues = inputValues * 0.8 + 0.1  # 0.1-0.9 aralığına getir
        
        # 2. İleri Yayılım
        for i, value in enumerate(inputValues[:len(layers[0])]):
            layers[0][i].value = value
        
        runAI()

        # 3. Hata ve Delta Hesaplama (GELİŞTİRİLMİŞ)
        output_layer = layers[-1]
        errors = []
        deltas = {}
        
        # Çıkış katmanı deltaları
        for i, neuron in enumerate(output_layer[:len(targetValues)]):
            error = targetValues[i] - neuron.value
            errors.append(error)
            
            # Aktivasyon türevi için güvenli hesaplama
            deriv = neuron.activation_derivative()
            if np.isnan(deriv) or deriv < 1e-8:
                deriv = 0.1  # Minimum türev değeri
            
            delta = error * deriv
            deltas[neuron.id] = delta

        # Gizli katman deltaları (geriye doğru)
        for layer_idx in reversed(range(len(layers)-1)):
            for neuron in layers[layer_idx]:
                delta_sum = 0
                
                # Sonraki katmandaki tüm bağlantıları tarayarak delta toplamını hesapla
                if neuron.id in connections[layer_idx]:
                    for conn in connections[layer_idx][neuron.id]:
                        to_id = conn.connectedTo[1]
                        if to_id in deltas:
                            delta_sum += conn.weight * deltas[to_id]
                
                # Aktivasyon türevi için güvenli hesaplama
                deriv = neuron.activation_derivative()
                if np.isnan(deriv) or deriv < 1e-8:
                    deriv = 0.1  # Minimum türev değeri
                
                delta = delta_sum * deriv
                
                # Delta değerini makul bir aralıkta tut
                if abs(delta) < 1e-8:
                    delta = np.sign(delta) * 1e-8  # Çok küçük değerleri sınırla
                elif abs(delta) > 1.0:
                    delta = np.sign(delta) * 1.0  # Çok büyük değerleri sınırla
                
                deltas[neuron.id] = delta

        # 4. Gradyan Hesaplama (DEBUG ÖZELLİKLİ)
        all_gradients = []
        grad_debug_info = []  # Hata ayıklama için
        
        for layer_idx in range(len(layers)-1):
            if layer_idx in connections:
                for src_id, conn_list in connections[layer_idx].items():
                    src_neuron = get_neuron_by_id(src_id)
                    if src_neuron:
                        for conn in conn_list:
                            to_id = conn.connectedTo[1]
                            if to_id in deltas:
                                grad = deltas[to_id] * src_neuron.value
                                
                                # Gradyanı kontrol et ve düzelt
                                if np.isnan(grad) or np.isinf(grad):
                                    grad = 0.01
                                elif abs(grad) < 1e-8:
                                    grad = np.sign(grad) * 1e-8
                                
                                all_gradients.append(abs(grad))
                                
                                # Hata ayıklama bilgisi topla
                                grad_debug_info.append({
                                    'from': src_id,
                                    'to': to_id,
                                    'grad': grad,
                                    'delta': deltas[to_id],
                                    'src_val': src_neuron.value,
                                    'weight': conn.weight
                                })

        # Gradyan çeşitliliğini kontrol et
        if len(all_gradients) > 0:
            max_grad = max(all_gradients)
            min_grad = min(all_gradients)
            avg_grad = np.mean(all_gradients)
            
            # Gradyanları normalize et (isteğe bağlı)
            if max_grad > max_grad_norm:
                scale_factor = max_grad_norm / max_grad
                for neuron_id in deltas:
                    deltas[neuron_id] *= scale_factor
        else:
            max_grad = 0.0
            min_grad = 0.0
            avg_grad = 0.0
            print("UYARI: Hiç gradyan hesaplanamadı!")

        # 5. Ağırlık Güncellemeleri (DEBUG ÖZELLİKLİ)
        num_updates = 0
        total_update = 0.0
        
        for layer_idx in range(len(layers)-1):
            if layer_idx in connections:
                for src_id, conn_list in connections[layer_idx].items():
                    src_neuron = get_neuron_by_id(src_id)
                    if src_neuron:
                        # En büyük gradyanı bul
                        max_grad_in_layer = 0
                        max_conn = None
                        for conn in conn_list:
                            to_id = conn.connectedTo[1]
                            if to_id in deltas:
                                current_grad = deltas[to_id] * src_neuron.value
                                if abs(current_grad) > abs(max_grad_in_layer):
                                    max_grad_in_layer = current_grad
                                    max_conn = conn
                        
                        # Bağlantıları güncelle
                        for conn in conn_list:
                            to_id = conn.connectedTo[1]
                            if to_id in deltas:
                                grad = deltas[to_id] * src_neuron.value
                                boost = boost_factor if conn == max_conn else 1.0
                                update_id = id(conn)
                                prev_update = previous_updates.get(update_id, 0)
                                
                                # Güncelleme değerini hesapla
                                update = learning_rate * boost * grad + momentum * prev_update
                                
                                # Güncellemeyi uygula
                                conn.weight += update
                                previous_updates[update_id] = update
                                
                                num_updates += 1
                                total_update += abs(update)

        # 6. Hata ve Debug Bilgileri
        output_values = [neuron.value for neuron in layers[-1][:len(targetValues)]]
        current_error = hata_payi(targetValues[:len(output_values)], output_values)
        
        print("\n=== EĞİTİM ÖZETİ ===")
        print(f"Hata: {current_error:.6f}")
        print(f"LR: {learning_rate:.6f}")
        print(f"Gradyanlar: Min {min_grad:.6f}, Max {max_grad:.6f}, Ort {avg_grad:.6f}")
        print(f"Toplam Ağırlık Güncellemesi: {total_update:.6f} ({num_updates} bağlantı)")
        
       
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
        self.complexity_penalty = 0.001  # Aşırı büyümeyi cezalandırma katsayısı
        self.performance_threshold = 0.1  # Yapısal değişiklik tetikleme eşiği
        self.error_history = []
        
    def adapt_network(self, current_error):
        """Ağın performansına göre yapısal değişiklikler yapar"""
        self.error_history.append(current_error)
        if len(self.error_history) < 5:  # En az 5 iterasyon bekleyelim
            return
        
        # Son 5 hatanın ortalaması
        avg_error = sum(self.error_history[-5:]) / 5
        
        # Hata artıyorsa veya yeterince azalmıyorsa
        if avg_error > min(self.error_history) + self.performance_threshold:
            self._modify_architecture(avg_error)
            
    def _modify_architecture(self, avg_error):
        """Ağ mimarisini avg_error'a göre akıllıca değiştiren operasyonlar"""

        # Hata seviyesine göre ağırlıklandırılmış action seçimi
        if avg_error > 0.5:  # Çok yüksek hata - agresif değişiklikler
            action_weights = [0.3, 0.3, 0.2, 0.2]  # Nöron/katman ekleme ağırlıklı
        elif avg_error > 0.2:  # Orta seviye hata - dengeli değişiklikler
            action_weights = [0.4, 0.1, 0.4, 0.1]  # Orijinal ağırlıklar
        else:  # Düşük hata - ince ayarlar
            action_weights = [0.2, 0.0, 0.6, 0.2]  # Bağlantı optimizasyonu ağırlıklı

        # Hatanın trendine göre ağırlıkları ayarla
        if len(self.error_history) > 5:
            last_errors = self.error_history[-5:]
            trend = (last_errors[-1] - last_errors[0]) / 5  # Hata trendi

            if trend > 0:  # Hata artıyorsa
                action_weights = [x*1.5 for x in action_weights[:2]] + [x*0.7 for x in action_weights[2:]]  # Yapısal değişikliklere ağırlık ver
            elif trend < -0.05:  # Hata belirgin şekilde azalıyorsa
                action_weights = [x*0.7 for x in action_weights[:2]] + [x*1.3 for x in action_weights[2:]]  # İnce ayarlara ağırlık ver

        # Normalize ağırlıklar (toplamı 1 yap)
        total = sum(action_weights)
        action_weights = [w/total for w in action_weights]

        # Rastgele bir değişiklik türü seç
        action = random.choices([0, 1, 2, 3], weights=action_weights)[0]

        if action == 0:  # Nöron ekle
            self._add_neuron_strategically()
        elif action == 1:  # Katman ekle
            self._add_layer_strategically()
        elif action == 2:  # Bağlantıları optimize et
            self._optimize_connections()
        else:  # Nöron/katman sil
            self._prune_network()

    def _add_neuron_strategically(self):
        """Hatanın en yüksek olduğu katmana nöron ekler"""
        # En aktif olmayan katmanı bul
        layer_idx = self._find_weakest_layer()
        
        # Yeni nöron için aktivasyon tipini rastgele seç
        activation_types = ['sigmoid', 'tanh', 'relu']
        new_neuron = Neuron(default_value=0.0, 
                          activation_type=defaultNeuronActivationType)
        
        add_neuron(self.layers, self.connections, layer_idx, new_neuron)
        print(f"Adaptif nöron eklendi: Katman {layer_idx}, ID {new_neuron.id}")

    def _add_layer_strategically(self):
        """Ağa yeni bir katman ekler"""
        if len(self.layers) >= 5:  # Maksimum 5 katman
            return
            
        # En çok hatanın olduğu iki katman arasına ekle
        layer_idx = self._find_weakest_connection_point()
        
        # Yeni katmanda nöron sayısını belirle (ortalama değer)
        neuron_count = (len(self.layers[layer_idx]) + len(self.layers[layer_idx+1])) // 2
        new_neurons = [Neuron(0.0) for _ in range(max(2, neuron_count))]
        
        addLayer(self.layers, self.connections, layer_idx+1, new_neurons)
        print(f"Adaptif katman eklendi: Indeks {layer_idx+1}")

    def _optimize_connections(self):
        """Bağlantı ağırlıklarını ve yapısını akıllıca optimize eder"""
        weak_connections = self._find_weak_connections(threshold=0.1)

        for (layer_idx, from_id, conn) in weak_connections:
            from_neuron = get_neuron_by_id(from_id)
            to_neuron = get_neuron_by_id(conn.connectedTo[1])

            # Bağlantının önemini hesapla (aktivasyon ve ağırlığın çarpımı)
            connection_strength = abs(from_neuron.value * conn.weight)

            # Katman bilgilerini al
            current_layer_size = len(self.layers[layer_idx])
            next_layer_size = len(self.layers[layer_idx+1])

            # Silme kararı için kriterler
            should_remove = (
                connection_strength < 0.01 and  # Çok zayıf bağlantı
                current_layer_size > 10 and     # Kaynak katman yeterince büyük
                next_layer_size > 10 and        # Hedef katman yeterince büyük
                random.random() < 0.3           #%30 ihtimalle silme
            )

            if should_remove:
                # Bağlantıyı güvenli şekilde kaldır
                self._safely_remove_connection(layer_idx, from_id, conn)
            else:
                # Ağırlığı akıllıca resetle
                self._smart_reset_connection(conn, from_neuron, to_neuron)

    def _safely_remove_connection(self, layer_idx, from_id, conn):
        """Bağlantıyı güvenli şekilde kaldırır"""
        try:
            self.connections[layer_idx][from_id].remove(conn)
            print(f"Bağlantı kaldırıldı: {from_id}→{conn.connectedTo[1]}")

            # Eğer nöronun başka bağlantısı kalmadıysa
            if not self.connections[layer_idx][from_id]:
                # Giriş katmanında değilse ve çıkış katmanına bağlı değilse
                if layer_idx > 0 and layer_idx < len(self.layers)-1:
                    del self.connections[layer_idx][from_id]
                    print(f"Nöron {from_id}'in tüm bağlantıları kaldırıldı")
        except Exception as e:
            print(f"Bağlantı kaldırma hatası: {str(e)}")

    def _smart_reset_connection(self, conn, from_neuron, to_neuron):
        """Ağırlığı akıllıca resetler"""
        # Aktivasyonlara göre yeni ağırlık aralığı belirle
        strength_factor = (abs(from_neuron.value) + abs(to_neuron.value)) / 2

        if strength_factor < 0.1:  # Çok zayıf aktivasyon
            new_weight = random.uniform(-0.5, 0.5)
        elif strength_factor > 0.9:  # Çok güçlü aktivasyon
            # Mevcut yönü koru ama büyüklüğü ayarla
            sign = 1 if conn.weight > 0 else -1
            new_weight = sign * random.uniform(0.5, 2.0)
        else:  # Orta seviye
            new_weight = random.uniform(-1.0, 1.0)

        conn.weight = new_weight
        print(f"Bağlantı güncellendi: {from_neuron.id}→{to_neuron.id} yeni ağırlık: {new_weight:.4f}")

    def _prune_network(self):
        """Az kullanılan nöronları ve katmanları temizler"""
        # En az aktif nöronları bul
        inactive_neurons = self._find_inactive_neurons(threshold=0.01)
        
        for neuron_id in inactive_neurons:
            removeNeuron(self.layers, self.connections, neuron_id)
            print(f"Az kullanılan nöron kaldırıldı: {neuron_id}")

        # Eğer katman çok küçülmüşse tamamen kaldır
        for i in range(1, len(self.layers)-1):
            if len(self.layers[i]) < 2:
                removeLayer(self.layers, self.connections, i)
                print(f"Küçük katman kaldırıldı: {i}")
                break

    # Yardımcı analiz fonksiyonları
    def _find_weakest_layer(self):
        """En düşük aktivasyon ortalamasına sahip katmanı bulur"""
        layer_activity = []
        for i, layer in enumerate(self.layers[1:-1]):  # Giriş/çıkış hariç
            avg_activation = sum(n.value for n in layer) / len(layer)
            layer_activity.append((i+1, avg_activation))
        
        return min(layer_activity, key=lambda x: x[1])[0]

    def _find_weak_connections(self, threshold=0.1):
        """Zayıf bağlantıları tespit eder"""
        weak_conns = []
        for layer_idx in self.connections:
            for from_id, conn_list in self.connections[layer_idx].items():
                for conn in conn_list:
                    from_neuron = get_neuron_by_id(from_id)
                    to_neuron = get_neuron_by_id(conn.connectedTo[1])
                    if abs(conn.weight * from_neuron.value) < threshold:
                        weak_conns.append((layer_idx, from_id, conn))
        return weak_conns

    def _find_inactive_neurons(self, threshold=0.05):
        """Belirli bir eşiğin altında aktivite gösteren nöronları bulur"""
        inactive = []
        for layer in self.layers[1:-1]:  # Giriş/çıkış hariç
            for neuron in layer:
                if abs(neuron.value) < threshold:
                    inactive.append(neuron.id)
        return inactive[:2]  # En fazla 2 nöron sil

    def _find_weakest_connection_point(self):
        """Katmanlar arası en zayıf bağlantı noktasını bulur"""
        connection_strengths = []
        for layer_idx in range(len(self.layers)-1):
            total_strength = 0
            for conn_list in self.connections[layer_idx].values():
                for conn in conn_list:
                    total_strength += abs(conn.weight)
            connection_strengths.append((layer_idx, total_strength))
        
        return min(connection_strengths, key=lambda x: x[1])[0]



def train_network(X_train, y_train, epochs=None, intelligenceValue=None):
    dynamic_manager = DynamicNetworkManager(layers, connections)
    avg_error = float('inf')  # Başlangıçta sonsuz hata
    epoch = 0
    
    while True:
        # Eğer epochs belirtilmişse ve epoch sayısı aşıldıysa dur
        if epochs is not None and epoch >= epochs:
            break
        # Eğer intelligenceValue belirtilmişse ve hata yeterince düştüyse dur
        if intelligenceValue is not None and avg_error <= intelligenceValue:
            print(f"\nHata {intelligenceValue} değerinin altına düştü! Eğitim durduruldu.")
            break
            
        total_error = 0
        for X, y in zip(X_train, y_train):
            # İleri yayılım
            for i, val in enumerate(X[:len(layers[0])]):
                layers[0][i].value = val
            runAI()
            
            # Hata hesapla
            output = [neuron.value for neuron in layers[-1][:len(y)]]
            error = hata_payi(y, output)
            total_error += error
            
            # Geri yayılım
            TrainFor(X, y, connections)
            
            # Ağ yapısını adapte et
            dynamic_manager.adapt_network(error)
        
        avg_error = total_error / len(X_train)
        epoch += 1
        print(f"Epoch {epoch}, Error: {avg_error:.4f}, Ağ Boyutu: {sum(len(l) for l in layers)} nöron")
        
        # Her 10 epoch'ta bir görselleştir
        if epoch % 10 == 0:
            visualize_network(layers, connections, refresh=True)
    
    print(f"\nEğitim tamamlandı! Son Hata: {avg_error:.4f}")

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

def toggle_drawing_mode(event):
    global drawing_mode
    drawing_mode = not drawing_mode
    if drawing_mode:
        print("Çizim modu aktif! Sol tıkla çiz, sağ tıkla temizle, 'p' ile tahmin et.")
    else:
        print("Çizim modu pasif.")

def handle_drawing_click(event):
    if not drawing_mode or event.inaxes != ax_drawing:
        return
    
    if event.button == MouseButton.LEFT:
        # Piksel koordinatlarını al
        x, y = int(event.xdata), int(event.ydata)
        
        # 3x3 alanı boya (daha kalın çizgi için)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= x+i < 28 and 0 <= y+j < 28:
                    drawing_data[y+j, x+i] = min(1.0, drawing_data[y+j, x+i] + 0.3)  # Kademeli boyama
        
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
    if not drawing_mode:
        return
    
    # Çizimi ağa ver
    flat_drawing = drawing_data.flatten()
    for i in range(len(layers[0])):
        layers[0][i].value = flat_drawing[i] if i < len(flat_drawing) else 0
    
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

# Çizim alanını oluştur
def setup_drawing_area():
    global ax_drawing, drawing_canvas
    
    fig = plt.figure(figsize=(10, 5))
    ax_drawing = fig.add_subplot(121)
    ax_drawing.set_title("Rakam Çiz (Sol tık: çiz, Sağ tık: temizle)")
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








# Terminal giriş döngüsü
while True:
    if cmd == "exit":
        break
        
    if cmd == "refresh()":
        clearGUI()
        runAI()
        visualize_network(layers, connections, refresh=True)
        print(getOutput())
    # Komut satırına yeni komut ekleyin
    elif cmd == "draw()":
        setup_drawing_area()
        cmd = ""  # Komut döngüsünü kesmemek için
    elif cmd == "visualize()":
        visualizeNetwork = not visualizeNetwork
        
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
            
    elif (cmd.startswith("getNeuronV(")and cmd.endswith(")")):
        args = cmd[11:-1].split(";")
        for i in args:
            neuron_id = int(i)
            neuron = get_neuron_by_id(neuron_id)
            if neuron:
                print(f"Nöron {neuron_id} Değer: {neuron.value:.4f} | Ağırlıklı Toplam: {neuron.weightedSum:.4f}")
                
    elif (cmd.startswith("changeInputRandomly(")and cmd.endswith(")")):
        for i in layers[0]:
            i.value=random.uniform(0.1, 1)
        print("Giriş katmanı rastgele değerlerle güncellendi")
        
    elif cmd.startswith("trainFor(") and cmd.endswith(")"):
        try:
            params = cmd[len("trainFor("):-1].split(';')
            input_values = list(map(float, params[0].split(',')))
            target_values = list(map(float, params[1].split(',')))

            lr = float(params[2]) if len(params) > 2 else 0.1
            boost = float(params[3]) if len(params) > 3 else 2.0
            momentum_val = float(params[4]) if len(params) > 4 else 0.9
            max_grad = float(params[5]) if len(params) > 5 else 1.0

            TrainFor(input_values, target_values, connections,
                    learning_rate=lr,
                    boost_factor=boost,
                    momentum=momentum_val,
                    max_grad_norm=max_grad)
            cmd = "refresh()"
            continue
        except Exception as e:
            traceback.print_exc()
            print("Hatalı komut formatı! Doğru kullanım:\ntrainFor(girişler;hedefler;lr;boost;momentum;max_grad)")
            
    elif cmd.startswith("trainFromFile("):
        try:
            params = cmd[len("trainFromFile("):-1].split(';')
            x,y=modeltrainingprogram.read_csv_file(params[0])
            train_network(x,y,epochs=int(params[1]))
            print(f"Eğitim tamamlandı. {len(x)} örnek işlendi.")
        except Exception as e:
            traceback.print_exc()
            print(f"Dosya okuma hatası: {str(e)}")
            
    elif cmd.startswith("set_input"):
        try:
            values = list(map(float, cmd.split()[1:]))
            for i, val in enumerate(values[:len(layers[0])]):
                layers[0][i].value = val
            print(f"Giriş katmanı güncellendi: {values}")
        except Exception as e:
            print(f"Hata: {e}")
            
    else:
        print("\nGeçerli komutlar:")
        print("- print_network(): Ağ yapısını terminalde göster")
        print("- get_connection(from_id,to_id): Bağlantı ağırlığını göster")
        print("- add_neuron(layer_idx,value): Yeni nöron ekle")
        print("- removeNeuron(id): Nöron sil")
        print("- addLayer(index;[nöronlar]): Katman ekle")
        print("- removeLayer(index): Katman sil")
        print("- changeW(from;to;weight): Ağırlık değiştir")
        print("- changeN(id;value): Nöron değeri değiştir")
        print("- trainFor(inputs;targets;[lr]): Eğitim yap")
        print("- trainFromFile(path_to_csv;epochs): Dosyadan eğitim yap")
        print("- refresh(): Ağı yenile")
        print("- exit: Çıkış")

    cmd = input("\nKomut girin: ")