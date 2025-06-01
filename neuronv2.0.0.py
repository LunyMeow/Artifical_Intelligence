import csv
import random
import sys
import time
import traceback
from typing import Dict, List, Optional
import matplotlib

import modeltrainingprogram
import os
if os.name == 'nt':
    print("Windows sistemi")
    matplotlib.use('TkAgg')
else:
    print("Linux veya MacOS sistemi")
    matplotlib.use('Agg')  # Bu satÄ±rÄ± plt importundan Ã¶nce koy


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable




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

import mplcursors  # Fare etkileÅŸimi iÃ§in



import signal
import json
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime





class Neuron:
    next_id = 0  # Global olarak artan ID deÄŸeri
    
    def __init__(self, default_value:float=0.0, activation_type=None):
        self.value = default_value

        self.id = Neuron.next_id  # Otomatik ID ata
        Neuron.next_id += 1  # Sonraki nÃ¶ron iÃ§in ID artÄ±r
        self.activation_type = activation_type
        self.output = 0.0  # Ã‡Ä±ktÄ± deÄŸeri, aktivasyon fonksiyonundan sonra hesaplanacak
        self.weightedSum = 0

    def activation(self, x):
        if self.activation_type == 'sigmoid':
            x = np.clip(x, -500, 500)  # x'i -500 ile 500 arasÄ±nda tut
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'linear':
        # identity fonksiyon, girdiÄŸi olduÄŸu gibi geÃ§irir
            return x
        elif self.activation_type == "doubleSigmoid":
            x = np.clip(x, -500, 500)
            base_sigmoid = 1 / (1 + np.exp(-x))
            return 3 * base_sigmoid - 1  # AralÄ±k: -1 ila 2


        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def activation_derivative(self):
        if self.activation_type == 'sigmoid':
            # Ã‡Ä±ktÄ±yÄ± 0.01-0.99 aralÄ±ÄŸÄ±na sÄ±kÄ±ÅŸtÄ±r
            safe_output = np.clip(self.output, 0.01, 0.99)
            return safe_output * (1 - safe_output)  # f'(x) = f(x)(1 - f(x))
        elif self.activation_type == 'tanh':
            return 1 - self.output ** 2  # f'(x) = 1 - f(x)^2
        elif self.activation_type == 'linear':
            return 1  # f(x) = x â†’ f'(x) = 1
        elif self.activation_type == "doubleSigmoid":
            scaled_output = (self.output + 1) / 3  # -1 ila 2 â†’ 0 ila 1
            scaled_output = np.clip(scaled_output, 0.01, 0.99)
            return 3 * (scaled_output * (1 - scaled_output))


        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def calculate_weighted_sum(self, layers, connections):
        weighted_sum = 0
        bias_sum = 0  # Bias toplamÄ±
        
        for layer_idx in range(len(layers) - 1):
            for prev_neuron in layers[layer_idx]:
                for conn in connections[layer_idx].get(prev_neuron.id, []):
                    if conn.connectedTo[1] == self.id:
                        
                        weighted_sum += prev_neuron.value * conn.weight
                        bias_sum += conn.bias  # BaÄŸlantÄ± bias'larÄ±nÄ± topla
        
        self.weightedSum = weighted_sum + bias_sum  # Bias'Ä± ekle
        self.value = self.activation(self.weightedSum)
        self.output = self.value
        return self.value




class Connection:
    def __init__(self, weight=0, connectedToArg=[0, 0], bias=0.1):  # VarsayÄ±lan bias=0.1
        self.weight = weight
        self.connectedTo = connectedToArg
        self.bias = bias  # Bias parametresi eklendi
    
    def update_weight(self, learning_rate, delta):
        self.weight += learning_rate * delta
        self.bias += learning_rate * delta * 0.1  # Bias da gÃ¼ncelleniyor



visualizeNetwork =False
debug = False  # Global debug deÄŸiÅŸkeni
enable_logging = False  # Loglama varsayÄ±lan olarak kapalÄ±
showMatplot = False
#cmd = "train_custom(veri.csv;2,5,2;0.0004)" #program baÅŸlar baÅŸlamaz Ã§alÄ±ÅŸacak ilk komut
#cmd="train_custom(output.csv;27,1,2;0.1;1;0.5)" #lr deÄŸeri bÃ¼yÃ¼dÃ¼kÃ§e model daha hÄ±zlÄ± Ã¶ÄŸreniyor fakat ani yÃ¼kseliiÅŸler ve alÃ§almalar daha fazla oluyor bu sorunu da dinamik deÄŸiÅŸen minimum istenen hata deÄŸeri Ã¶zelliÄŸi Ã§Ã¶zÃ¼yor yani ÅŸuanda iyi Ã¶ÄŸrenim iÃ§in yÃ¼ksek lr deÄŸeri daha iyi
#cmd = "train_custom(2025-05-13_11-38-32_ASELS.IS_Training.csv;27,1,2;0.1;2;2)"
#cmd = "loadModel(trainingDatas/network_20250513_113946.pkl.gz)"
#Test :fakat lr deÄŸerini bÃ¼yÃ¼k veriler iÃ§in test etmedim bÃ¼yÃ¼k verilerde kÃ¼Ã§Ã¼k lr deÄŸeri gerekebilir 
#Ekleme Ã–nerisi :lr deÄŸerinin update_learning_rate fonksiyonunda nekadar azalÄ±p nekadar alÃ§almasÄ±nÄ± kontrol eden parametre factor deÄŸeri bu deÄŸer veriye baÄŸlÄ± olarak dinamik olarak deÄŸiÅŸirse daha iyi olabilir





# AÄŸ oluÅŸturma
randomMinWeight = -2.0
randomMaxWeight = 2.0



activation_types = ['sigmoid', 'tanh','linear','doubleSigmoid']
defaultNeuronActivationType='tanh'
defaultOutActivation = 'sigmoid'






# Ã–nce boÅŸ bir layers listesi oluÅŸtur
layers = []



# BaÄŸlantÄ±larÄ± oluÅŸturma

connections = {}
def setConnections(preserve_weights=True):
    global layers, connections
    
    # Eski aÄŸÄ±rlÄ±klarÄ± sakla
    old_weights = {}
    if preserve_weights:
        for layer_idx in connections:
            for neuron_id in connections[layer_idx]:
                for conn in connections[layer_idx][neuron_id]:
                    key = (layer_idx, neuron_id, conn.connectedTo[1])
                    old_weights[key] = conn.weight
    
    # Yeni baÄŸlantÄ±larÄ± oluÅŸtur
    new_connections = {layer_idx: {} for layer_idx in range(len(layers) - 1)}
    for layer_idx in range(len(layers) - 1):
        for neuron in layers[layer_idx]:
            for next_neuron in layers[layer_idx + 1]:
                key = (layer_idx, neuron.id, next_neuron.id)
                
                if preserve_weights and key in old_weights:
                    # Eski aÄŸÄ±rlÄ±ÄŸÄ± koru
                    weight = old_weights[key]
                else:
                    # Yeni aÄŸÄ±rlÄ±k oluÅŸtur
                    weight = random.uniform(-1/np.sqrt(len(layers[layer_idx])), 
                                    1/np.sqrt(len(layers[layer_idx])))
                
                conn = Connection(connectedToArg=[neuron.id, next_neuron.id], weight=weight)
                
                if neuron.id not in new_connections[layer_idx]:
                    new_connections[layer_idx][neuron.id] = []
                new_connections[layer_idx][neuron.id].append(conn)
    
    connections = new_connections

def setLayers(neuronInLayers):
    """KatmanlarÄ± ve nÃ¶ron sayÄ±larÄ±nÄ± ayarlar"""
    global layers  # Global layers listesini kullanacaÄŸÄ±mÄ±zÄ± belirtiyoruz
    layers.clear()  # Ã–nceki katmanlarÄ± temizle
    
    for layerIndex,neuronCount in enumerate(neuronInLayers):

        # Her katman iÃ§in yeni nÃ¶ron listesi oluÅŸtur
        layer = [
            Neuron(
                default_value=1,
                activation_type=defaultOutActivation if layerIndex == len(neuronInLayers) - 1 else "tanh"
            )
            for _ in range(neuronCount)
        ]

        layers.append(layer)
    
    setConnections(preserve_weights=False)


def createFileName(symbol=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"trainingDatas/{symbol}_network_{timestamp}.pkl.gz"





def save_network_optimized(filename=None,symbol=""):
    if filename is None:
        filename = createFileName(symbol=symbol)
    """YÃ¼ksek performanslÄ± binary kayÄ±t fonksiyonu"""
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
    """YÃ¼ksek performanslÄ± binary yÃ¼kleme fonksiyonu"""
    global layers, connections, Neuron
    
    with gzip.open(filename, 'rb') as f:
        network_data = pickle.load(f)
    
    # Global deÄŸiÅŸkenleri gÃ¼ncelle
    (randomMinWeight, randomMaxWeight, defaultNeuronActivationType, 
     visualizeNetwork, debug) = network_data['config']
    Neuron.next_id = network_data['next_id']
    
    # KatmanlarÄ± yeniden oluÅŸtur
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
    
    # BaÄŸlantÄ±larÄ± yeniden oluÅŸtur (defaultdict ile hÄ±zlÄ± eriÅŸim)
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
    Bir deÄŸeri kaynak aralÄ±ktan hedef aralÄ±ÄŸa Ã¶lÃ§eklendirir.

    :param x: DÃ¶nÃ¼ÅŸtÃ¼rÃ¼lecek deÄŸer
    :param source_min: Kaynak aralÄ±ÄŸÄ±n alt sÄ±nÄ±rÄ±
    :param source_max: Kaynak aralÄ±ÄŸÄ±n Ã¼st sÄ±nÄ±rÄ±
    :param target_min: Hedef aralÄ±ÄŸÄ±n alt sÄ±nÄ±rÄ±
    :param target_max: Hedef aralÄ±ÄŸÄ±n Ã¼st sÄ±nÄ±rÄ±
    :return: Ã–lÃ§eklendirilmiÅŸ deÄŸer
    """
    return target_min + ((x - source_min) / (source_max - source_min)) * (target_max - target_min)


def runAI():
    for layer in layers[1:]:
        for neuron in layer:
            #print(f"NÃ¶ron {neuron.id}: {neuron.value}")
            neuron.calculate_weighted_sum(layers,connections)
    #print(f"Son deÄŸer: {scale_value(get_neuron_by_id(30).value,0,1,0,8)}")
    lastNeuronValues =[]
    for neuron in layers[-1]:
        lastNeuronValues.append(neuron.value)
    return lastNeuronValues



# Global deÄŸiÅŸkenler
global win, plot, scatter, lines, app
win = None
plot = None
scatter = None
lines = []
app = None





def visualize_network(layers, connections, node_size=20,refresh=False,DONTVisualize=False):
    if DONTVisualize:
        return
    if not visualizeNetwork:
        return
    global win, plot, scatter, lines, app

    if win is None or not refresh:
        if QtWidgets.QApplication.instance() is None:
            app = QtWidgets.QApplication([])
        else:
            app = QtWidgets.QApplication.instance()
        
        win = pg.GraphicsLayoutWidget(show=True,size=(1200,800))
        win.setWindowTitle("Sinir AÄŸÄ± GÃ¶rselleÅŸtirme")
        plot = win.addPlot()
        #view = plot.getViewBox()
        plot.hideAxis('bottom')
        plot.hideAxis('left')
        #view.setAspectLocked(True)
        win.setBackground("#dedede")
    else:
        plot.clear()
        for line in lines:
            plot.removeItem(line)
        lines.clear()
        if scatter:
            plot.removeItem(scatter)
    #app = QtWidgets.QApplication([])
    #win = pg.GraphicsLayoutWidget(show=True, size=(1200, 800))
    #win.setWindowTitle("Sinir AÄŸÄ± GÃ¶rselleÅŸtirme (Scatter + Lines Hybrid)")
    
    
    #plot = win.addPlot()

    view = plot.getViewBox()
    view.setMouseEnabled(x=True, y=True)  # YakÄ±nlaÅŸtÄ±rma ve kaydÄ±rma aktif
    view.setAspectLocked(False)  # Oran kilidi kapalÄ± (serbest zoom)
    view.setMenuEnabled(False)  # SaÄŸ tÄ±k menÃ¼sÃ¼ kapalÄ±
    view.wheelZoomFactor = 1.1  # Zoom hassasiyeti

    # 1. NÃ¶ron pozisyonlarÄ± ve renkleri
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
            
            # Aktivasyona gÃ¶re renk (kÄ±rmÄ±zÄ±: dÃ¼ÅŸÃ¼k, yeÅŸil: yÃ¼ksek)
            norm_val = max(0, min(1, neuron.value))  # 0-1 arasÄ±na sÄ±kÄ±ÅŸtÄ±r
            node_colors.append(pg.mkColor(
                int(255 * (1 - norm_val)),  # R
                int(255 * norm_val),        # G
                0,                         # B
                200                        # Alpha
            ))
    


    # 2. BaÄŸlantÄ± verilerini topla
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
                    print(f"UyarÄ±: GeÃ§ersiz baÄŸlantÄ± {conn.connectedTo}")

    # 3. NÃ¶ronlarÄ± Ã§iz (ScatterPlot)
    scatter = pg.ScatterPlotItem(
        pos=np.array(list(pos.values())),
        size=node_size,
        brush=node_colors,
        pen=pg.mkPen('k', width=1),
        pxMode=True
    )
    plot.addItem(scatter)

    # 4. BaÄŸlantÄ±larÄ± Ã§iz (LineCollection benzeri yaklaÅŸÄ±m)
    if edges and edge_weights:
        # AÄŸÄ±rlÄ±klarÄ± normalize et
        min_w, max_w = min(edge_weights), max(edge_weights)
        range_w = max_w - min_w if max_w != min_w else 1
        
        # Her baÄŸlantÄ± iÃ§in ayrÄ± Ã§izgi oluÅŸtur
        for i, (u, v) in enumerate(edges):
            # AÄŸÄ±rlÄ±ÄŸa gÃ¶re stil belirle
            norm_w = (edge_weights[i] - min_w) / range_w
            
            if norm_w < 0.5:  # Negatif aÄŸÄ±rlÄ±k (mavi)
                intensity = int(255 * (1 - 2*norm_w))
                color = pg.mkColor(0, 0, 255, intensity)
                width = 1 + 3 * (1 - 2*norm_w)
            else:  # Pozitif aÄŸÄ±rlÄ±k (kÄ±rmÄ±zÄ±)
                intensity = int(255 * (2*(norm_w - 0.5)))
                color = pg.mkColor(255, 0, 0, intensity)
                width = 1 + 3 * (2*(norm_w - 0.5))
            
            # Ã‡izgiyi ekle
            line = pg.PlotDataItem(
                x=[pos[u][0], pos[v][0]],
                y=[pos[u][1], pos[v][1]],
                pen=pg.mkPen(color, width=width)
            )
            plot.addItem(line)

    # 5. NÃ¶ron etiketleri
    for neuron_id, (x, y) in pos.items():
        text = pg.TextItem(str(neuron_id), color='w', anchor=(0.5, 0.5))
        text.setFont(QtGui.QFont('Arial', max(8, node_size//3)))
        text.setPos(x, y)
        view.addItem(text)

    # 6. Tooltip iÅŸlevselliÄŸi
    tooltip = pg.TextItem(color='k', anchor=(0, 1), fill=(255, 255, 255, 200))
    edge_tooltip = pg.TextItem(color='k', anchor=(0.5, 0.5), fill=(255, 255, 255, 200))
    view.addItem(tooltip)
    view.addItem(edge_tooltip)
    tooltip.hide()
    edge_tooltip.hide()

    def on_hover(event):
        mouse_pos = view.mapSceneToView(event)
        
        # Ã–nce baÄŸlantÄ±larÄ± kontrol et
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
            edge_tooltip.setText(f"BaÄŸlantÄ±: {u} â†’ {v}\nAÄŸÄ±rlÄ±k: {weight:.6f}")
            edge_pos = (np.array(pos[u]) + np.array(pos[v])) / 2
            edge_tooltip.setPos(edge_pos[0], edge_pos[1])
            edge_tooltip.show()
            tooltip.hide()
        else:
            edge_tooltip.hide()
            
            # Sonra nÃ¶ronlarÄ± kontrol et
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
                
                text = (f"NÃ¶ron {closest_node}\n"
                       f"DeÄŸer: {neuron.value:.4f}\n"
                       f"Toplam: {neuron.weightedSum:.4f}\n"
                       f"\nGelen baÄŸlantÄ±lar ({len(incoming)}):\n")
                
                text += "\n".join(f"  {conn['from']} â†’ {conn['weight']:.4f}" for conn in incoming[:5])  # Ä°lk 5
                if len(incoming) > 5: text += "\n  ..."
                
                text += f"\n\nGiden baÄŸlantÄ±lar ({len(outgoing)}):\n"
                text += "\n".join(f"  â†’ {conn['to']} ({conn['weight']:.4f})" for conn in outgoing[:5])
                if len(outgoing) > 5: text += "\n  ..."
                
                tooltip.setText(text)
                tooltip.setPos(mouse_pos.x(), mouse_pos.y())
                tooltip.show()
            else:
                tooltip.hide()
    
    view.scene().sigMouseMoved.connect(on_hover)

    # 7. GÃ¶sterge (legend)
    legend = pg.LegendItem(offset=(10, 10), size=(150, 100))
    legend.setParentItem(view)
    legend.addItem(pg.PlotDataItem(pen=pg.mkPen((255,0,0), width=3)), "Pozitif aÄŸÄ±rlÄ±k")
    legend.addItem(pg.PlotDataItem(pen=pg.mkPen((0,0,255), width=3)), "Negatif aÄŸÄ±rlÄ±k")
    legend.addItem(pg.PlotDataItem(brush=pg.mkBrush((255,100,100))), "DÃ¼ÅŸÃ¼k aktivasyon")
    legend.addItem(pg.PlotDataItem(brush=pg.mkBrush((100,255,100))), "YÃ¼ksek aktivasyon")

    # 8. Performans ayarlarÄ±
    plot.setMenuEnabled(False)
    
    if not refresh:
        win.show()
        app.exec_()
    else:
        win.show()
        QtWidgets.QApplication.processEvents()






def change_weight(connections, from_id, to_id, new_weight):
    """
    Belirli bir baÄŸlantÄ±nÄ±n aÄŸÄ±rlÄ±ÄŸÄ±nÄ± deÄŸiÅŸtirir.

    :param connections: Katmanlar arasÄ± baÄŸlantÄ±lar
    :param from_id: BaÄŸlantÄ±dan gelen nÃ¶ronun ID'si
    :param to_id: BaÄŸlantÄ±ya giden nÃ¶ronun ID'si
    :param new_weight: Yeni aÄŸÄ±rlÄ±k
    """
    # connections dict'si Ã¼zerinden gezerek doÄŸru baÄŸlantÄ±yÄ± bulalÄ±m
    for layer_connections in connections.values():
        for neuron_id, conn_list in layer_connections.items():
            for conn in conn_list:
                # BaÄŸlantÄ± [from_id, to_id] olup olmadÄ±ÄŸÄ±nÄ± kontrol et
                if conn.connectedTo == [from_id, to_id]:
                    conn.weight = new_weight  # Yeni aÄŸÄ±rlÄ±ÄŸÄ± gÃ¼ncelle
                    print(f"BaÄŸlantÄ± gÃ¼ncellendi: {from_id} -> {to_id} yeni weight: {new_weight}")
                    return  # Ä°ÅŸlem tamamlandÄ±ÄŸÄ±nda fonksiyonu sonlandÄ±r

    print(f"Hata: {from_id} ile {to_id} arasÄ±nda baÄŸlantÄ± bulunamadÄ±.")  # BaÄŸlantÄ± bulunamazsa hata mesajÄ±














def get_neuron_by_id(neuron_id, layersArg=layers):
    for layer in layersArg:
        for neuron in layer:
            if neuron.id == neuron_id:
                return neuron
    return None  # EÄŸer nÃ¶ron bulunamazsa None dÃ¶ndÃ¼r




# Hata payÄ± fonksiyonu
def hata_payi(target, output):
    # Listeleri numpy dizilerine dÃ¶nÃ¼ÅŸtÃ¼r
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
    AÄŸdaki baÄŸlantÄ± bilgilerini dÃ¶ndÃ¼rÃ¼r.
    
    Parametreler:
    - layer_idx: Belirli bir katmanÄ±n baÄŸlantÄ±larÄ±nÄ± getir (None ise tÃ¼m katmanlar)
    - detailed: DetaylÄ± bilgi (kimden kime) ekler
    
    DÃ¶nÃ¼ÅŸ DeÄŸeri:
    - EÄŸer detailed=False: {layer_idx: {from_id: [weight1, weight2, ...]}}
    - EÄŸer detailed=True: {layer_idx: [(from_id, to_id, weight), ...]}
    """
    result = {}
    
    # TÃ¼m katmanlar iÃ§in
    if layer_idx is None:
        target_layers = connections.keys()
    else:
        if layer_idx not in connections:
            print(f"UyarÄ±: {layer_idx}. katman bulunamadÄ±!")
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
            # DetaylÄ± format: [(from_id, to_id, weight), ...]
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
# KULLANIM Ã–RNEKLERÄ°:

# 1. TÃ¼m baÄŸlantÄ±larÄ± basit ÅŸekilde alma
all_connections = get_connections()
print("TÃ¼m baÄŸlantÄ±lar (basit):", all_connections)

# 2. Belirli bir katmanÄ±n baÄŸlantÄ±larÄ±nÄ± detaylÄ± alma
layer_1_conns = get_connections(layer_idx=1, detailed=True)
print("\n1. katman baÄŸlantÄ±larÄ± (detaylÄ±):")
for conn in layer_1_conns[1]:
    print(f"{conn[0]} -> {conn[1]} | AÄŸÄ±rlÄ±k: {conn[2]:.4f}")"""

# 3. Belirli bir nÃ¶ronun baÄŸlantÄ±larÄ±nÄ± bulma
def get_neuron_connections(neuron_id, incoming=True, outgoing=True):
    """
    Returns list of Connection objects instead of tuples
    """
    found = []
    
    # Ã–nce nÃ¶ronun hangi katmanda olduÄŸunu bulalÄ±m
    neuron_layer = None
    for layer_idx, layer in enumerate(layers):
        for neuron in layer:
            if neuron.id == neuron_id:
                neuron_layer = layer_idx
                break
        if neuron_layer is not None:
            break
    
    if neuron_layer is None:
        print(f"UyarÄ±: {neuron_id} ID'li nÃ¶ron bulunamadÄ±!")
        return found
    
    # Gelen baÄŸlantÄ±lar (Ã¶nceki katmandan)
    if incoming and neuron_layer > 0:
        prev_layer_idx = neuron_layer - 1
        if prev_layer_idx in connections:
            for from_id, conn_list in connections[prev_layer_idx].items():
                for conn in conn_list:
                    if conn.connectedTo[1] == neuron_id:
                        found.append(conn)  # Return the Connection object itself
    
    # Giden baÄŸlantÄ±lar (sonraki katmana)
    if outgoing and neuron_layer < len(layers) - 1:
        current_layer_idx = neuron_layer
        if current_layer_idx in connections and neuron_id in connections[current_layer_idx]:
            for conn in connections[current_layer_idx][neuron_id]:
                found.append(conn)  # Return the Connection object itself
    
    return found






def print_error_progress(current_error, target_error=0.01, width=50):
    """Hata deÄŸerine gÃ¶re ilerleme Ã§ubuÄŸu"""
    if target_error is None or target_error <= 0:
        target_error = 0.01  # VarsayÄ±lan deÄŸer
    
    progress = min(1.0, max(0.0, 1 - (current_error / target_error)))
    filled = int(progress * width)
    bar = '[' + '=' * filled + ' ' * (width - filled) + ']'
    print(f"\rHata Modu: {current_error:.6f} {bar} %{progress*100:.1f}", end='')
    if progress >= 0.99:
        print(f"\nHedef hata deÄŸerine ulaÅŸÄ±ldÄ±: {current_error:.6f} <= {target_error:.6f}")

def print_epoch_progress(current_epoch, total_epochs, current_error, width=50):
    """Epoch sayÄ±sÄ±na gÃ¶re ilerleme Ã§ubuÄŸu"""
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




# Global deÄŸiÅŸkenler
error_history = []
epoch_history = []
learning_rate_history=[]
start_time = None


stopEpoch = False #ctrl C yapÄ±nca eÄŸitimi durdurmasÄ± iÃ§in


def signal_handler(sig, frame):
    """Ctrl+C ile Ã§Ä±kÄ±ÅŸ yakalandÄ±ÄŸÄ±nda Ã§aÄŸrÄ±lacak fonksiyon"""
    global enable_logging,stopEpoch
    
    print("\nEÄŸitim durduruldu.")
    stopEpoch =True
    if enable_logging:
        print("Veriler kaydediliyor...")
        
        #visualize_saved_errors(save_and_plot_errors())

    

import os

def klasor_hazirla(yol):
    """Verilen yol iÃ§in klasÃ¶r yapÄ±sÄ±nÄ± hazÄ±rlar"""
    try:
        os.makedirs(yol, exist_ok=True)
        print(f"KlasÃ¶r yapÄ±sÄ± hazÄ±r: {yol}")
        return True
    except Exception as e:
        print(f"Hata oluÅŸtu: {e}")
        return False


def save_and_plot_errors():
    """Hata geÃ§miÅŸini kaydet ve gÃ¶rselleÅŸtir"""
    global error_history, epoch_history, start_time
    
    if not error_history:
        print("Kaydedilecek veri yok.")
        return
    
    # Ã‡Ä±ktÄ± dosyasÄ±nÄ±n adÄ±nÄ± belirle - sadece bir dosya oluÅŸtur
    outputFolder="trainingDatas/"
    klasor_hazirla(outputFolder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_base = f"training_errors_{timestamp}"
    output_file_json = f"{output_file_base}.json"
    output_file_png = f"{output_file_base}.png"
    
    # Toplam eÄŸitim sÃ¼resini hesapla
    total_time = time.time() - start_time if start_time else 0
    
    # Veriyi JSON formatÄ±nda kaydet
    data = {
        "errors": error_history,
        "epochs": epoch_history,
        "total_time_seconds": total_time,
        "final_error": error_history[-1] if error_history else None,
        "learning_rates": learning_rate_history  # Bu yeni eklenen kÄ±sÄ±m
    }
    
    with open(outputFolder+output_file_json, 'w') as f:
        json.dump(data, f)
    
    print(f"Hata verileri {output_file_json} dosyasÄ±na kaydedildi.")
    
    
    
    
    return outputFolder+output_file_json

def testModel(testFile: str, inputNum: int, targetNum: int,DONTVisualize=False):
    X, y = modeltrainingprogram.read_csv_file(testFile)
    fail = 0
    success = 0
    if debug or enable_logging:
        targetsAndOutputs=[]
    
    for a, i in enumerate(X):
        cmd_set_input(i)
        cmd_refresh(DONTVisualize=DONTVisualize)
        output, maxIndex, _ = getOutput()
        if debug or enable_logging: targetsAndOutputs.append([output[0],y[a][0]])
        if targetNum == 1:
            # Tek bir Ã§Ä±ktÄ± varsa, kÃ¼Ã§Ã¼k bir toleransla eÅŸitlik kontrolÃ¼ yapÄ±lÄ±r
            if abs(output[0] - y[a][0]) < 0.3:  # tolerans isteÄŸe baÄŸlÄ± ayarlanabilir
                success += 1
            else:
                fail += 1
        else:
            # Ã‡oklu Ã§Ä±kÄ±ÅŸta, en yÃ¼ksek skora sahip indeks kontrolÃ¼
            if y[a][maxIndex] == max(y[a]):
                success += 1
            else:
                fail += 1

    total = success + fail
    accuracy = (success / total) * 100 if total > 0 else 0
    if debug or enable_logging:
        for b in targetsAndOutputs:
            print(b)
    print(f"BaÅŸarÄ±lÄ±: {success}, BaÅŸarÄ±sÄ±z: {fail}, DoÄŸruluk: %{accuracy:.2f}")
    return accuracy






def train_network(X_train, y_train, batch_size=1, epochs=None, intelligenceValue=None, learning_rate=0.05,useDynamicModelChanges=True,symbol="",epochNumberForLimitError=None):
    global error_history, epoch_history, start_time, enable_logging, learning_rate_history
    
    if enable_logging:
        print("Hata grafiÄŸi kaydÄ± etkin. EÄŸitim sonunda grafik oluÅŸturulacak.")
    
    error_history = []
    epoch_history = []
    learning_rate_history = []
    newLR = 0.0
    
    signal.signal(signal.SIGINT, signal_handler)
    
    cortical_column = CorticalColumn(learning_rateArg=learning_rate, targetError=epochs if epochs <1 else None,
                                     maxEpochForTargetError=8000 if epochNumberForLimitError==None else epochNumberForLimitError,
                                     originalNetworkModel=[len(liste) for liste in layers],useDynamicModelChanges=useDynamicModelChanges)

    
    avg_error = float('inf')
    epoch = 0
    total_samples = len(X_train)
    start_time = time.time()

    if len(layers[0]) != len(X_train[0]):
        print(f"UyarÄ±: GiriÅŸ boyutu uyumsuz! AÄŸ giriÅŸi: {len(layers[0])}, Veri giriÅŸi: {len(X_train[0])}")
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
                
                

                #sys.stdout.write(f"\nEpoch {epoch+1}/{epochs if epochs is not None else 'âˆ'} - Ä°lerleme: {processed_samples}/{total_samples} ({100*processed_samples/total_samples:.1f}%) - Ortalama Hata: {avg_error:.6f}")
                #sys.stdout.flush()
                if debug:
                    print(f"\nEpoch {epoch+1}/{epochs if epochs is not None else 'âˆ'} - Ä°lerleme: {processed_samples}/{total_samples} ({100*processed_samples/total_samples:.1f}%)")
                    print(f"Ortalama Hata: {avg_error:.6f}")



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
        print(f"\n=== EÄÄ°TÄ°M TAMAMLANDI ===")
        print(f"Toplam SÃ¼re: {total_time/60:.1f} dakika | Toplam saniye: {total_time:.3f}")
        print(f"Son Hata: {avg_error:.6f}")
        print(f"Toplam Epoch: {epoch}")
        print(f"Final AÄŸ YapÄ±sÄ±: {[len(layer) for layer in layers]}")

        try:
            filename =createFileName(symbol=symbol)
            save_network_optimized(filename,symbol=symbol)
            print(f"AÄŸ yapÄ±sÄ± {filename} dosyasÄ±na kaydedildi")
        except Exception as a:
            print("Modeli dosyaya kaydetme sÄ±rasÄ±nda hata meydana geldi.")
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
    """KaydedilmiÅŸ hata verilerini geliÅŸmiÅŸ grafiklerle gÃ¶rselleÅŸtir"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    errors = np.array(data["errors"])
    epochs = np.array(data["epochs"])
    learning_rates = np.array(data.get("learning_rates", [0.05]*len(epochs)))
    
    # Ana grafik
    plt.figure(figsize=(15, 10))
    
    # 1. Hata eÄŸrisi (ana grafik)
    ax1 = plt.subplot(2, 2, (1, 3))  # 2 satÄ±r, 2 sÃ¼tun, 1 ve 3'Ã¼ birleÅŸtir
    main_plot = plt.plot(epochs, errors, 'b-', linewidth=1, label='Ortalama Hata')
    scatter1 = plt.plot(epochs, errors, 'ro', markersize=1)[0]
    
    # EÄŸilim Ã§izgisi ekleme
    z = np.polyfit(epochs, errors, 3)
    p = np.poly1d(z)
    trend_line = plt.plot(epochs, p(epochs), "r--", linewidth=2, label='EÄŸilim Ã‡izgisi')[0]
    
    # DÃ¶nÃ¼m noktalarÄ±nÄ± bulma
    diff = np.diff(errors)
    turning_points = np.where(np.diff(np.sign(diff)))[0] + 1
    
    if len(turning_points) > 0:
        for tp in turning_points:
            plt.plot(epochs[tp], errors[tp], 'go', markersize=2, label='DÃ¶nÃ¼m NoktasÄ±' if tp == turning_points[0] else "")
    
    # Grafik Ã¶zelleÅŸtirme
    plt.title('EÄŸitim SÄ±rasÄ±nda Hata DeÄŸiÅŸimi ve EÄŸilimi')
    plt.xlabel('Epoch')
    plt.ylabel('Ortalama Hata')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Son 20% epoch iÃ§in yakÄ±nlaÅŸtÄ±rÄ±lmÄ±ÅŸ grafik
    ax2 = plt.subplot(2, 2, 2)
    last_20 = int(len(epochs) * last_20Arg)
    line2 = plt.plot(epochs[last_20:], errors[last_20:], 'b-', linewidth=1.5)[0]
    scatter2 = plt.plot(epochs[last_20:], errors[last_20:], 'ro', markersize=2)[0]
    
    # Son bÃ¶lÃ¼m iÃ§in lineer regresyon
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        epochs[last_20:], errors[last_20:])
    reg_line = plt.plot(epochs[last_20:], intercept + slope*epochs[last_20:], 
             'g--', linewidth=2, 
             label=f'EÄŸim: {slope:.2e}\nRÂ²: {r_value**2:.2f}')[0]
    
    plt.title(f'Son %{int(100-last_20Arg*100)} Epoch YakÄ±nlaÅŸtÄ±rma')
    plt.xlabel('Epoch')
    plt.ylabel('Hata')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Learning Rate DeÄŸiÅŸimi GrafiÄŸi
    ax3 = plt.subplot(2, 2, 4)
    lr_line = plt.plot(epochs, learning_rates, 'm-', linewidth=1.5, label='Learning Rate')[0]
    lr_scatter = plt.plot(epochs, learning_rates, 'co', markersize=2)[0]
    
    # Learning rate iÃ§in eÄŸilim Ã§izgisi
    z_lr = np.polyfit(epochs, learning_rates, 1)
    p_lr = np.poly1d(z_lr)
    plt.plot(epochs, p_lr(epochs), "k--", linewidth=1, label=f'EÄŸilim: {z_lr[0]:.2e}x + {z_lr[1]:.2f}')
    
    plt.title('Learning Rate DeÄŸiÅŸimi')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Genel bilgiler
    stats_text = (
        f"BaÅŸlangÄ±Ã§ Hata: {errors[0]:.6f}\n"
        f"Son Hata: {errors[-1]:.6f}\n"
        f"En DÃ¼ÅŸÃ¼k Hata: {np.min(errors):.6f}\n"
        f"Ortalama Hata: {np.mean(errors):.6f}\n"
        f"Standart Sapma: {np.std(errors):.6f}\n"
        f"DÃ¶nÃ¼m NoktalarÄ±: {len(turning_points)}\n"
        f"BaÅŸlangÄ±Ã§ LR: {learning_rates[0]:.6f}\n"
        f"Son LR: {learning_rates[-1]:.6f}\n"
        f"Toplam Epoch: {len(epochs)}\n"
        f"Toplam SÃ¼re: {data['total_time_seconds']:.2f} sn"
    )
    
    plt.figtext(0.75, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.5), 
                fontsize=9)
    
    plt.tight_layout()
    
    # GrafiÄŸi kaydet
    output_file = os.path.splitext(filename)[0] + "_advanced_viz.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"GeliÅŸmiÅŸ hata grafiÄŸi {output_file} dosyasÄ±na kaydedildi.")
    
    # EÄŸilim analizi
    analyze_trend(errors, epochs, last_20Arg=last_20Arg)
    
    # Fare etkileÅŸimi ekleme
    def format_annotation(sel):
        x, y = sel.target
        epoch = int(x)
        if sel.artist in [scatter1, scatter2]:  # Hata grafiklerindeki noktalar
            error = y
            sel.annotation.set_text(f"Epoch: {epoch}\nHata: {error:.6f}")
        elif sel.artist == lr_scatter:  # Learning rate grafiÄŸindeki noktalar
            lr = y
            sel.annotation.set_text(f"Epoch: {epoch}\nLearning Rate: {lr:.6f}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
    
    # TÃ¼m grafikler iÃ§in cursor ekle
    crs1 = mplcursors.cursor([scatter1, scatter2, lr_scatter], hover=True)
    crs1.connect("add", format_annotation)
    if showMatplot:
        if debug:
            plt.show(block=True)
        else:
            plt.show()

def analyze_trend(errors, epochs,last_20Arg):
    """Hata eÄŸilimini analiz eder ve yorumlar"""
    # Son %(default 20)'lik kÄ±sÄ±m iÃ§in eÄŸim analizi

    last_20 = int(len(epochs) * last_20Arg)
    slope, _, _, _, _ = stats.linregress(epochs[last_20:], errors[last_20:])
    
    print("\n=== HATA EÄÄ°LÄ°M ANALÄ°ZÄ° ===")
    print(f"Son hata deÄŸeri: {errors[-1]:.6f}")
    print(f"Son %{100-last_20Arg*100} epoch'taki ortalama hata eÄŸimi: {slope:.2e}")
    
    if slope > 1e-6:
        print("UYARI: Hatalarda artÄ±ÅŸ eÄŸilimi var! Model overfitting olabilir veya Ã¶ÄŸrenme oranÄ± yÃ¼ksek olabilir.")
    elif slope < -1e-6:
        print("Hatalarda dÃ¼ÅŸÃ¼ÅŸ eÄŸilimi devam ediyor. EÄŸitime devam edilebilir.")
    else:
        print("Hatalar sabitlenmiÅŸ gÃ¶rÃ¼nÃ¼yor. Daha fazla eÄŸitimin faydasÄ± olmayabilir.")
    
    # YakÄ±nsama kontrolÃ¼
    last_10_errors = errors[-10:]
    std_last_10 = np.std(last_10_errors)
    if std_last_10 < 0.001:
        print(f"Hatalar yakÄ±nsamÄ±ÅŸ (son 10 epoch std: {std_last_10:.6f})")
    else:
        print(f"Hatalar henÃ¼z tam yakÄ±nsamadÄ± (son 10 epoch std: {std_last_10:.6f})")
    
    # Ã–neriler
    print("\n=== Ã–NERÄ°LER ===")
    if slope > 0 and len(errors) > 50:
        print("- Ã–ÄŸrenme oranÄ±nÄ± azaltmayÄ± deneyin")
        print("- Regularization ekleyin")
        print("- Early stopping uygulayÄ±n")
    elif slope < -1e-4:
        print("- Model hala Ã¶ÄŸreniyor, eÄŸitime devam edebilirsiniz")
    else:
        print("- Model performansÄ±nÄ± artÄ±rmak iÃ§in mimariyi deÄŸiÅŸtirmeyi deneyin")
    
    return slope #hatalardaki artÄ±ÅŸ eÄŸimi







def getOutput():
    output_values = []
    max_value = -1
    max_index = -1
    
    # TÃ¼m Ã§Ä±ktÄ± nÃ¶ronlarÄ±nÄ± iÅŸle
    for i, neuron in enumerate(layers[-1]):
        value = neuron.value
        weighted_sum = neuron.weightedSum
        
        # En yÃ¼ksek aktivasyonu takip et
        if value > max_value:
            max_value = value
            max_index = i
        
        output_values.append(value)
    
    # En yÃ¼ksek aktivasyon bilgisini ekle

    
    
    
    return output_values,max_index,max_value














def evaluate_network(X_test, y_test):
    """Modeli deÄŸerlendir"""
    correct = 0
    for i in range(len(X_test)):
        # GiriÅŸi ayarla
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
            print(f"Test {i}/{len(X_test)} - DoÄŸruluk: {correct/(i+1):.2%}")
    
    print(f"\nFinal Test DoÄŸruluk: {correct/len(X_test):.2%}")



bias_is=None
def disable_all_biases():
    global bias_is
    bias_is=False
    for layer_idx in connections:
        for neuron_id in connections[layer_idx]:
            for conn in connections[layer_idx][neuron_id]:
                conn.bias = 0

# KullanÄ±m:
disable_all_biases()

def enable_all_biases():
    global bias_is
    bias_is=True
    for layer_idx in connections:
        for neuron_id in connections[layer_idx]:
            for conn in connections[layer_idx][neuron_id]:
                conn.bias = random.uniform(-0.1, 0.1)  # Rastgele kÃ¼Ã§Ã¼k deÄŸerlerle yeniden baÅŸlat

# KullanÄ±m:
#enable_all_biases()



"""
    [BaÅŸla]
       â”‚
       â”œâ”€â–º (Opsiyonel) BaÅŸlangÄ±Ã§ Checkpoint'u OluÅŸtur  âœ…
       â”‚       â””â”€ Modelin baÅŸlangÄ±Ã§ durumunu kaydet (rollback iÃ§in)
       â”‚
       â”œâ”€â–º Backpropagation Ã‡alÄ±ÅŸtÄ±r  âœ…
       â”‚
       â”œâ”€â–º NÃ¶ron SaÄŸlÄ±ÄŸÄ±nÄ± Hesapla  âœ…
       â”‚       â””â”€ Her nÃ¶ronun aktivasyon, aÄŸÄ±rlÄ±k daÄŸÄ±lÄ±mÄ± ve Ã¶ÄŸrenme potansiyeli gibi metriklerini deÄŸerlendir
       â”‚            (AyrÄ±ca nÃ¶ron saÄŸlÄ±ÄŸÄ± puanlarÄ±nÄ± kaydet ve izleyerek hangi nÃ¶ronlarÄ±n zayÄ±f olduÄŸunu belirle)
       â”‚
       â”œâ”€â–º Kritik NÃ¶ronlarÄ± Belirle ve DÃ¼zenle  
       â”‚       â””â”€ ZayÄ±f nÃ¶ronlar iÃ§in:
       â”‚             â”œâ”€ Gerekirse nÃ¶ron ekle (daha fazla Ã¶ÄŸrenme kapasitesi iÃ§in)
       â”‚             â”œâ”€ Gerekirse nÃ¶ron sil (fazla gÃ¼rÃ¼ltÃ¼ veya zayÄ±f performans iÃ§in)
       â”‚             â””â”€ Gerekirse nÃ¶ronlarÄ± birleÅŸtir veya bÃ¶l
       â”‚             â””â”€ (Deneysel modda yapÄ±lan deÄŸiÅŸikliklerde geÃ§ici ekleme/silme uygulayarak performans karÅŸÄ±laÅŸtÄ±rmasÄ± yap)
       â”‚
       â”œâ”€â–º Katman SaÄŸlÄ±ÄŸÄ±nÄ± Ã–lÃ§  
       â”‚       â””â”€ Katman bazlÄ± performans ve overfitting kontrolÃ¼ yap
       â”‚            (Her katmanÄ±n validasyon performansÄ±nÄ± da izleyerek geniÅŸleme/silme kararÄ±nÄ± destekle)
       â”‚
       â”œâ”€â–º Katman YapÄ±sÄ±nÄ± Dinamik Olarak DÃ¼zenle  
       â”‚       â””â”€ Gerekirse yeni katman ekle veya mevcut katmanlarÄ± sil
       â”‚            (YapÄ±sal geniÅŸlemenin overfitting'e yol aÃ§Ä±p aÃ§madÄ±ÄŸÄ±nÄ± kontrol etmek iÃ§in test/simÃ¼lasyon uygulayÄ±n)
       â”‚
       â”œâ”€â–º DeÄŸiÅŸiklikleri Kaydet  
       â”‚       â””â”€ YapÄ±sal ve hiperparametre deÄŸiÅŸikliklerinin kaydÄ±nÄ± (checkpoint) oluÅŸtur  
       â”‚            (Her Ã¶nemli deÄŸiÅŸiklikten sonra modelin durumunu kaydedin)
       â”‚
       â”œâ”€â–º Hiperparametreleri Dinamik Olarak Ayarla  
       â”‚       â””â”€ Ã–zellikle learning rate ve regularization parametrelerini, loss trendine ve nÃ¶ron/katman saÄŸlÄ±k metriklerine gÃ¶re artÄ±r/azalt
       â”‚            (Entegre kontrol algoritmasÄ±yla, Ã¶rneÄŸin â€œpatienceâ€ ve â€œstability windowâ€ kullanarak karar verin)
       â”‚
       â”œâ”€â–º Kademeli ve Ã–lÃ§Ã¼lebilir Karar MekanizmasÄ± Uygula  
       â”‚       â””â”€ DeÄŸiÅŸiklikleri her epochâ€™da deÄŸil, belirli bir sÃ¼re boyunca (Ã¶rneÄŸin, birkaÃ§ epoch) izleyerek
       â”‚             â€¢ GeÃ§ici dalgalanmalarÄ±n etkisinden kaÃ§Ä±nmak iÃ§in iyileÅŸme eÅŸiÄŸi (threshold) belirle
       â”‚             â€¢ Rollback zaman penceresini tanÄ±mla
       â”‚
       â”œâ”€â–º Belirli Bir SÃ¼re (Stability Window) Bekle ve GÃ¶zlemle  âœ…
       â”‚       â””â”€ YapÄ±lan deÄŸiÅŸikliklerin etkisini birkaÃ§ epoch boyunca deÄŸerlendir  
       â”‚             (Bu sÃ¼re zarfÄ±nda modelin performansÄ±, nÃ¶ron ve katman saÄŸlÄ±ÄŸÄ± metrikleri detaylÄ±ca izlenir)
       â”‚
       â”œâ”€â–º SimÃ¼lasyon ve A/B Testleri Uygula (Opsiyonel)  
       â”‚       â””â”€ GeliÅŸtirilen dinamik yapÄ± dÃ¼zenleme algoritmasÄ±nÄ± kÃ¼Ã§Ã¼k Ã¶lÃ§ekli testlerle dene
       â”‚             â€¢ FarklÄ± stratejilerin performansÄ±nÄ± karÅŸÄ±laÅŸtÄ±rarak en etkili olanlarÄ± seÃ§
       â”‚
       â””â”€â–º Hata ArttÄ± mÄ±?  
               â”œâ”€â–º Evet â†’ Geri Al (Rollback: Ã–nceki Checkpoint'e dÃ¶n ve deÄŸiÅŸiklikleri geri Ã§ek)
               â””â”€â–º HayÄ±r â†’ [Devam]
    """

def is_multiple(a, b):
    return a % b == 0 if b != 0 else False

def detect_u_turn(loss_history, check_window=20, delta=1e-4):
    """
    Loss deÄŸerlerinin Ã¶nce dÃ¼ÅŸtÃ¼ÄŸÃ¼nÃ¼ sonra yÃ¼kseldiÄŸini tespit eder.
    
    Args:
        loss_history (list of float): KayÄ±tlÄ± loss deÄŸerleri.
        check_window (int): U-dÃ¶nÃ¼ÅŸÃ¼ kontrol etmek iÃ§in geÃ§miÅŸteki kaÃ§ epochâ€™a bakÄ±lacak.
        delta (float): Azalma ve artma algÄ±lama hassasiyeti.

    Returns:
        bool: EÄŸer U-dÃ¶nÃ¼ÅŸÃ¼ varsa True, yoksa False.
        dict: Ek bilgiler (minimum nokta, hangi epochâ€™ta oldu, ortalama deÄŸiÅŸimler).
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


def progress_bar(current, total=50, bar_length=40):
    percent = int(100 * current / total)
    filled = int(bar_length * percent // 100)
    bar = '#' * filled + '-' * (bar_length - filled)
    sys.stdout.write(f"\r[{bar}] {percent}%  ({current}/{total})")
    sys.stdout.flush()

class CorticalColumn:
    
    def __init__(self, log_file="network_changes.log", learning_rateArg=0.3,targetError=None,
                 maxEpochForTargetError=1000,originalNetworkModel=None,
                 overfit_threshold=0.9,useDynamicModelChanges=True):
        global layers, connections
        
        self.firstLearningRate = learning_rateArg
        self.learningRate = learning_rateArg
        self.neuronHealtThreshould = 0.3
        self.change_log = []  # DeÄŸiÅŸiklik loglarÄ±nÄ± tutacak liste
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
        if debug and self.targetError != None:print(f"Hedef hata deÄŸerine eriÅŸilememesi durumunda minimum hata deÄŸerinin sÄ±fÄ±rlanmasÄ± iÃ§in gereken epoch sayÄ±sÄ± {self.maxEpochForTargetError} deÄŸerinin katlarÄ± olarak belirlendi.")
        self.originalNetworkModel = originalNetworkModel
        self.neuron_health_history = {}  # Dictionary to store health history of neurons

                # Overfitting control
        self.overfit_threshold = overfit_threshold
        self.val_error_history = []
        self.useDynamicModelChanges = useDynamicModelChanges

        
        

        # Log dosyasÄ±nÄ± baÅŸlat (varsa sil, yenisini oluÅŸtur)
        with open(self.log_file, 'w') as f:
            f.write("")  # BoÅŸ dosya oluÅŸtur

    def _get_timestamp(self):
        """GeÃ§erli zaman damgasÄ±nÄ± ISO formatÄ±nda dÃ¶ndÃ¼rÃ¼r"""
        return datetime.now().isoformat()
    
    def _append_to_log(self, entry):
        """Log giriÅŸini hem bellekte hem de dosyaya ekler"""
        self.change_log.append(entry)
        with open(self.log_file, 'a') as f:
            json.dump(entry, f)
            f.write("\n")

    def log_change(self, change_type, details):
        global layers
        """DeÄŸiÅŸiklikleri loglayan yardÄ±mcÄ± fonksiyon"""
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
        Belirtilen epoch aralÄ±ÄŸÄ±ndaki hata deÄŸerlerinin eÄŸimini hesaplar

        Parametreler:
        error_history (list): Hata deÄŸerlerinin listesi
        start_epoch (int): BaÅŸlangÄ±Ã§ epoch indeksi (None ise son %20'nin baÅŸlangÄ±cÄ±)
        end_epoch (int): BitiÅŸ epoch indeksi (None ise son epoch)

        Returns:
        float: Hata eÄŸimi (pozitif = hatalar artÄ±yor, negatif = hatalar azalÄ±yor)
        float: RÂ² deÄŸeri (eÄŸimin gÃ¼venilirliÄŸi, 1'e yakÄ±n = gÃ¼venilir)
        """
        if not error_history or len(error_history) < 2:
            return 0.0, 0.0

        # VarsayÄ±lan aralÄ±klarÄ± ayarla
        if start_epoch is None:
            start_epoch = int(len(error_history) * 0.8)  # Son %20'nin baÅŸlangÄ±cÄ±
        if end_epoch is None:
            end_epoch = len(error_history) - 1  # Son epoch

        # GeÃ§erli aralÄ±ÄŸÄ± kontrol et
        start_epoch = max(0, min(start_epoch, len(error_history)-1))
        end_epoch = max(0, min(end_epoch, len(error_history)-1))

        if start_epoch >= end_epoch:
            return 0.0, 0.0

        # SeÃ§ilen aralÄ±ktaki hata ve epoch deÄŸerlerini al
        selected_errors = error_history[start_epoch:end_epoch+1]
        epochs = list(range(start_epoch, end_epoch+1))

        # Lineer regresyon ile eÄŸim ve RÂ² deÄŸerini hesapla
        slope, intercept, r_value, _, _ = stats.linregress(epochs, selected_errors)
        r_squared = r_value**2

        return slope, r_squared

    def monitor_network(self, avg_error,eveyXEpochAdaptNetwork=10):
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

        if slope<1e-6 and slope >-1e-6 and debug:
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
            # Apply a small automatic decay to prevent stagnation (0.9995^1000 â‰ˆ 0.61, so gradual)
            # This helps ensure the learning rate doesn't stay too high for too long
            
            # Get new learning rate based on loss trend
            new_lr = self.update_learning_rate(self.learningRate, self.loss_history,slopeArg=slope)
            
            # If learning rate changed significantly, reset cooldown counter

            self.learningRate = new_lr
            self.last_lr_change_epoch = self.current_epoch


        if self.current_epoch % eveyXEpochAdaptNetwork == 2 :  # Her 10 epoch'ta bir
            self.log_change('avg_error_debug', {
                'avg_error': avg_error,

            })

            progress_bar(self.loss_history[1]-avg_error,total=self.loss_history[1])
            
            if self.useDynamicModelChanges:
                self.adapt_network_structure(avg_error)
            

        
        return self.learningRate,None
    
    def update_learning_rate(self, current_lr, loss_history, 
                             patience=60, # Increased from 20
                             min_lr=0.1, max_lr=4.0,  # Reduced max_lr
                             factor=0.0005,  # Reduced from 0.002
                             threshold=1e-2, increase_threshold=5e-2,slopeArg=None):  # More conservative increase threshold
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
        if abs(slopeArg) > 0.1:  # Ani hata deÄŸiÅŸimlerinde
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
        if len(loss_history)>=100:
            if abs(loss_history[-1]-loss_history[-100]) < 0.0002 :
                new_lr=0.7
                self.log_change('slope is close to 0 lr_up',{'slope':slopeArg,'old_lr':current_lr,'new_lr':new_lr,'last 1 and last 100 error different':abs(loss_history[-1]-loss_history[-100])})
                return min(new_lr,max_lr)
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
            
            if debug:print(f"Error increasing! Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
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
                if debug:print(f"Min limit reached. Learning rate slightly increased: {current_lr:.6f} -> {new_lr:.6f}")
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
                    
                if debug:print(f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f} (improvement: {improvement:.6f})")
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
                if debug:print(f"Max limit reached. Learning rate decreased: {current_lr:.6f} -> {new_lr:.6f}")
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
                    
                if debug:print(f"Learning rate increased: {current_lr:.6f} -> {new_lr:.6f} (improvement: {improvement:.6f})")
                return new_lr
        
        # Case 3: Moderate improvement - maintain current learning rate with tiny decay
        if debug:
            print(f"Learning rate only slightly decayed: {current_lr:.6f} -> {decayed_lr:.6f} (improvement: {improvement:.6f})")
        # Mevcut mantÄ±ÄŸÄ± cosine decay ile birleÅŸtir
        return max(decayed_lr * 0.99, min_lr)  # Ekstra yavaÅŸ decay

    def backpropagation(self,input_data, target_data):
        global layers, connections

        # 1. Ä°leri Besleme - GiriÅŸ verilerini aÄŸa ver
        # GiriÅŸ katmanÄ±ndaki nÃ¶ron deÄŸerlerini ayarla
        for i, value in enumerate(input_data):
            if i < len(layers[0]):
                layers[0][i].value = value

        # Ä°leri besleme iÅŸlemi - tÃ¼m aÄŸÄ± hesapla

        runAI()

        # 2. Hata Hesaplama
        # Ã‡Ä±kÄ±ÅŸ katmanÄ±ndaki her nÃ¶ron iÃ§in hata hesapla
        output_layer = layers[-1]
        output_errors = []

        for i, neuron in enumerate(output_layer):
            if i < len(target_data):
                error = target_data[i] - neuron.value
                output_errors.append(error)
            else:
                output_errors.append(0)  # Hedef veri yoksa hata 0

        # 3. Geri YayÄ±lÄ±m
        # Her katman iÃ§in delta deÄŸerlerini hesapla (Ã§Ä±kÄ±ÅŸtan giriÅŸe doÄŸru)
        deltas = [[] for _ in range(len(layers))]

        # Ã–nce Ã§Ä±kÄ±ÅŸ katmanÄ±ndaki delta deÄŸerlerini hesapla
        for i, neuron in enumerate(output_layer):
            if i < len(output_errors):
                # Delta = Hata * Aktivasyon fonksiyonunun tÃ¼revi
                delta = output_errors[i] * neuron.activation_derivative()
                deltas[-1].append(delta)
            else:
                deltas[-1].append(0)

        # Gizli katmanlar iÃ§in delta deÄŸerlerini hesapla (geriye doÄŸru)
        for layer_idx in range(len(layers)-2, 0, -1):  # Son gizli katmandan ilk gizli katmana
            for i, neuron in enumerate(layers[layer_idx]):
                error = 0
                # Bu nÃ¶rondan sonraki katmana olan tÃ¼m baÄŸlantÄ±larÄ± kontrol et
                for conn in connections[layer_idx].get(neuron.id, []):
                    # Sonraki katmandaki nÃ¶ronu bul
                    next_layer_idx = layer_idx + 1
                    for j, next_neuron in enumerate(layers[next_layer_idx]):
                        if conn.connectedTo[1] == next_neuron.id:
                            # Bu baÄŸlantÄ±nÄ±n aÄŸÄ±rlÄ±ÄŸÄ± * sonraki nÃ¶ronun deltasÄ±
                            error += conn.weight * deltas[next_layer_idx][j]
                            break
                        
                # Delta = Hata * Aktivasyon fonksiyonunun tÃ¼revi
                delta = error * neuron.activation_derivative()
                deltas[layer_idx].append(delta)

        # 4. AÄŸÄ±rlÄ±k GÃ¼ncelleme
        for layer_idx in range(len(layers)-1):
            for i, neuron in enumerate(layers[layer_idx]):
                for conn in connections[layer_idx].get(neuron.id, []):
                    # Bir sonraki katmandaki baÄŸlÄ± nÃ¶ronu bul
                    next_layer_idx = layer_idx + 1
                    for j, next_neuron in enumerate(layers[next_layer_idx]):
                        if conn.connectedTo[1] == next_neuron.id:
                            # AÄŸÄ±rlÄ±k deÄŸiÅŸimi = Ã¶ÄŸrenme oranÄ± * delta * nÃ¶ron Ã§Ä±ktÄ±sÄ±
                            
                            if debug:
                                weight_change = self.learningRate * deltas[next_layer_idx][j] * neuron.value
                                old_weight = conn.weight
                                #print("debug values :",weight_change,self.learningRate,deltas[next_layer_idx][j],neuron.value)
                            # Mevcut Connection sÄ±nÄ±fÄ±nÄ±zdaki update_weight metodunu kullan
                            conn.update_weight(self.learningRate, deltas[next_layer_idx][j] * neuron.value)
                            break
                        
        # Toplam hata deÄŸerini hesapla ve dÃ¶ndÃ¼r (MSE - Mean Squared Error)
        total_error = sum(error**2 for error in output_errors) / len(output_errors) if output_errors else 0
        


        if debug:
            # AÄŸÄ±rlÄ±k gÃ¼ncellemelerini logla
            weight_updates = []
            for layer_idx in range(len(layers)-1):
                for i, neuron in enumerate(layers[layer_idx]):
                    for conn in connections[layer_idx].get(neuron.id, []):
                        weight_updates.append({
                            'from_neuron': conn.connectedTo[0],
                            'to_neuron': conn.connectedTo[1],
                            'old_weight': old_weight,  # GÃ¼ncellemeden Ã¶nceki aÄŸÄ±rlÄ±k
                            'new_weight': conn.weight,
                            'change': conn.weight-old_weight
                        })

            if weight_updates:
                self.log_change('weight_updates', {
                    'count': len(weight_updates),
                    'average_change': sum(abs(w['change']) for w in weight_updates) / len(weight_updates),
                    'updates': weight_updates[:10]  # Ä°lk 10 gÃ¼ncellemeyi gÃ¶ster (performans iÃ§in)
                })

        return total_error
        
    def adapt_neurons(self):
        pass
                
    
    def calculate_neuron_health(self,neuron):
        global layers,connections
        # 1. NÃ¶ronun aktivasyon deÄŸeri
        activation_score = neuron.output

        # 2. BaÄŸlantÄ±larÄ±n aÄŸÄ±rlÄ±k deÄŸerlerini kontrol et
        weight_sum = 0
        weight_count = 0

        # Gelen baÄŸlantÄ±larÄ± bul
        for layer_idx in range(len(layers) - 1):
            for prev_neuron in layers[layer_idx]:
                for conn in connections[layer_idx].get(prev_neuron.id, []):
                    if conn.connectedTo[1] == neuron.id:
                        weight_sum += abs(conn.weight)  # Mutlak deÄŸer kullan
                        weight_count += 1

        # Giden baÄŸlantÄ±larÄ± bul (eÄŸer bu nÃ¶ron Ã§Ä±kÄ±ÅŸ katmanÄ±nda deÄŸilse)
        current_layer_idx = None
        for layer_idx, layer in enumerate(layers):
            if any(n.id == neuron.id for n in layer):
                current_layer_idx = layer_idx
                break
            
        if current_layer_idx is not None and current_layer_idx < len(layers) - 1:
            for conn in connections.get(current_layer_idx, {}).get(neuron.id, []):
                weight_sum += abs(conn.weight)
                weight_count += 1

        # Ortalama aÄŸÄ±rlÄ±k (eÄŸer baÄŸlantÄ± varsa)
        avg_weight = weight_sum / weight_count if weight_count > 0 else 0

        # 3. Aktivasyon tÃ¼revi - nÃ¶ronun ne kadar Ã¶ÄŸrenmeye aÃ§Ä±k olduÄŸunu gÃ¶sterir
        learning_potential = neuron.activation_derivative()

        # TÃ¼m faktÃ¶rleri birleÅŸtirerek bir saÄŸlÄ±k puanÄ± hesapla
        # Bu formÃ¼lÃ¼ kendi ihtiyaÃ§larÄ±nÄ±za gÃ¶re ayarlayabilirsiniz
        health_score = (
            0.4 * activation_score +  # Aktivasyon deÄŸerine %40 aÄŸÄ±rlÄ±k ver
            0.4 * avg_weight +        # Ortalama aÄŸÄ±rlÄ±ÄŸa %40 aÄŸÄ±rlÄ±k ver
            0.2 * learning_potential  # Ã–ÄŸrenme potansiyeline %20 aÄŸÄ±rlÄ±k ver
        )

        return health_score
    

        


    def find_neuron_layer(self,neuron_id):
        """Verilen ID'ye sahip nÃ¶ronun hangi katmanda olduÄŸunu bulur"""
        global layers
        for layer_idx, layer in enumerate(layers):
            for neuron in layer:
                if neuron.id == neuron_id:
                    return layer_idx,neuron
        return None

    def add_neuron_to_layer(self, layer_index=None, neuron_id=None):
        """
        Belirli bir katmana yeni bir nÃ¶ron ekler ve sadece bu nÃ¶ronla ilgili baÄŸlantÄ±larÄ± oluÅŸturur

        Args:
            layer_index: NÃ¶ronun ekleneceÄŸi katmanÄ±n indeksi (None ise neuron_id'nin katmanÄ± kullanÄ±lÄ±r)
            neuron_id: Referans nÃ¶ron ID'si (bunun katmanÄ±na yeni nÃ¶ron eklenir, layer_index None ise)
        """
        global layers, connections

        # EÄŸer layer_index verilmediyse ve neuron_id verildiyse, nÃ¶ronun katmanÄ±nÄ± bul
        if layer_index is None and neuron_id is not None:
            layer_index = self.find_neuron_layer(neuron_id)
            if layer_index is None:
                if debug:print(f"Hata: ID'si {neuron_id} olan nÃ¶ron bulunamadÄ±.")
                return None

        # Hala layer_index belirlenemedi ise, varsayÄ±lan olarak son gizli katmanÄ± kullan
        if layer_index is None:
            if len(layers) <= 2:  # Sadece giriÅŸ ve Ã§Ä±kÄ±ÅŸ katmanlarÄ± varsa
                layer_index = 0  # GiriÅŸ katmanÄ±na ekle (veya tercih ettiÄŸiniz baÅŸka bir strateji)
            else:
                layer_index = len(layers) - 2  # Son gizli katman (Ã§Ä±kÄ±ÅŸ katmanÄ±ndan bir Ã¶nceki)

            if debug:
                print(f"Katman indeksi belirlenmediÄŸi iÃ§in varsayÄ±lan olarak katman {layer_index} kullanÄ±lÄ±yor.")

        if layer_index < 0 or layer_index >= len(layers):
            print(f"Hata: {layer_index} indeksi geÃ§erli bir katman indeksi deÄŸil.")
            return None

        # EÄŸer eklenen nÃ¶ron Ã§Ä±kÄ±ÅŸ katmanÄ±na aitse linear, deÄŸilse default aktivasyon
        activation_type = defaultOutActivation if layer_index == len(layers) - 1 else defaultNeuronActivationType

        # Yeni nÃ¶ron oluÅŸtur
        new_neuron = Neuron(activation_type=activation_type)

        # Yeni nÃ¶ronu katmana ekle
        layers[layer_index].append(new_neuron)

        # Ã–nceki katmandan bu nÃ¶rona baÄŸlantÄ±lar oluÅŸtur
        if layer_index > 0:
            prev_layer_idx = layer_index - 1
            for prev_neuron in layers[prev_layer_idx]:
                weight = random.uniform(-1 / np.sqrt(len(layers[layer_index])), 
                                        1 / np.sqrt(len(layers[layer_index])))
                conn = Connection(connectedToArg=[prev_neuron.id, new_neuron.id], weight=weight)

                if prev_neuron.id not in connections[prev_layer_idx]:
                    connections[prev_layer_idx][prev_neuron.id] = []

                connections[prev_layer_idx][prev_neuron.id].append(conn)

        # Bu nÃ¶rondan sonraki katmana baÄŸlantÄ±lar oluÅŸtur
        if layer_index < len(layers) - 1:
            for next_neuron in layers[layer_index + 1]:
                weight = random.uniform(-1 / np.sqrt(len(layers[layer_index])), 
                                        1 / np.sqrt(len(layers[layer_index])))
                conn = Connection(connectedToArg=[new_neuron.id, next_neuron.id], weight=weight)

                if new_neuron.id not in connections[layer_index]:
                    connections[layer_index][new_neuron.id] = []

                connections[layer_index][new_neuron.id].append(conn)

        if debug:
            print(f"Katman {layer_index}'e yeni nÃ¶ron (ID: {new_neuron.id}) eklendi.")

        return new_neuron


    def remove_neuron_from_layer(self,layer_index=None, neuron_id=None):
        """
        Belirli bir nÃ¶ronu ve sadece onunla ilgili baÄŸlantÄ±larÄ± siler

        Args:
            layer_index: NÃ¶ronun bulunduÄŸu katmanÄ±n indeksi (None ise otomatik bulunur)
            neuron_id: Silinecek nÃ¶ronun ID'si
        """
        global layers, connections

        if neuron_id is None:
            print("Hata: Silinecek nÃ¶ronun ID'si belirtilmedi.")
            return False

        # EÄŸer layer_index verilmediyse, nÃ¶ronun katmanÄ±nÄ± bul
        if layer_index is None:
            layer_index = self.find_neuron_layer(neuron_id)
            if layer_index is None:
                print(f"Hata: ID'si {neuron_id} olan nÃ¶ron bulunamadÄ±.")
                return False

        if layer_index < 0 or layer_index >= len(layers):
            print(f"Hata: {layer_index} indeksi geÃ§erli bir katman indeksi deÄŸil.")
            return False

        # NÃ¶ronu bul
        neuron_to_remove = None
        for i, neuron in enumerate(layers[layer_index]):
            if neuron.id == neuron_id:
                neuron_to_remove = neuron
                neuron_index = i
                break
            
        if neuron_to_remove is None:
            print(f"Hata: ID'si {neuron_id} olan nÃ¶ron, katman {layer_index}'de bulunamadÄ±.")
            return False

        # NÃ¶ronu katmandan Ã§Ä±kar
        layers[layer_index].pop(neuron_index)

        # Ã–nceki katmandan bu nÃ¶rona gelen baÄŸlantÄ±larÄ± sil
        if layer_index > 0:
            prev_layer_idx = layer_index - 1
            for prev_neuron_id in list(connections[prev_layer_idx].keys()):
                connections[prev_layer_idx][prev_neuron_id] = [
                    conn for conn in connections[prev_layer_idx][prev_neuron_id] 
                    if conn.connectedTo[1] != neuron_id
                ]

        # Bu nÃ¶rondan sonraki katmana giden baÄŸlantÄ±larÄ± sil
        if layer_index < len(layers) - 1:
            if neuron_id in connections[layer_index]:
                del connections[layer_index][neuron_id]

        if debug:
            print(f"Katman {layer_index}'den nÃ¶ron (ID: {neuron_id}) silindi.")

        return True


    
    def adapt_network_structure(self, avg_train_error, avg_val_error=None):
        """
        Dinamik yapÄ± adaptasyonu:
        - Overfitting kontrolÃ¼ ve gerekirse nÃ¶ron kÄ±rpma
        - NÃ¶ron seviyesinde optimizasyon
        - Katman seviyesinde optimizasyon
        - Stratejik bÃ¼yÃ¼me
        """
        # 0. Overfitting kontrolÃ¼
        if avg_val_error is not None:
            self.val_error_history.append(avg_val_error)
            gap = avg_val_error - avg_train_error
            if gap > self.overfit_threshold:
                self.prune_for_overfitting(gap)
                return

        # 1. NÃ¶ron seviyesinde optimizasyon
        self.neuron_level_optimization(avg_train_error)
        # 2. Katman seviyesinde optimizasyon
        self.layer_level_optimization(avg_train_error)
        # 3. Stratejik bÃ¼yÃ¼me
        self.strategic_growth(avg_train_error)

    def prune_for_overfitting(self, gap):
        """
        AÅŸÄ±rÄ± Ã¶ÄŸrenme tespit edilirse, nÃ¶ron sayÄ±sÄ±nÄ± azaltarak basitleÅŸtirir
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
                # SaÄŸlÄ±k skoru dÃ¼ÅŸÃ¼k nÃ¶ronlarÄ± Ã¶ncelikli kaldÄ±r
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
    
    def calculate_optimal_layer_sizes(self, input_size, output_size):
        """
        GiriÅŸ ve Ã§Ä±kÄ±ÅŸ boyutuna gÃ¶re optimal hidden layer boyutlarÄ±nÄ± hesaplar
        """
        # Temel kural: Hidden layer boyutu giriÅŸ ve Ã§Ä±kÄ±ÅŸÄ±n ortalamasÄ±ndan bÃ¼yÃ¼k olmalÄ±
        # ama Ã§ok bÃ¼yÃ¼k olmamalÄ±
        avg_size = (input_size + output_size) / 2
        min_size = max(input_size, output_size) * 1.3  # En az giriÅŸ/Ã§Ä±kÄ±ÅŸÄ±n 1.5 katÄ±
        max_size = input_size * 3  # GiriÅŸin 3 katÄ±nÄ± geÃ§memeli

        # Ã‡ok bÃ¼yÃ¼k giriÅŸler iÃ§in (500+ gibi) farklÄ± kurallar
        if input_size > 100:
            min_size = input_size * 1.2  # GiriÅŸin %20 fazlasÄ±
            max_size = input_size * 2    # GiriÅŸin 2 katÄ±

        return {
            'min_hidden': int(min_size),
            'max_hidden': int(max_size),
            'recommended': int(min(max_size, max(min_size, avg_size * 1.5)))
        }

    def detect_excessive_neurons(self, layer_idx):
        """
        Belirli bir katmandaki fazla nÃ¶ronlarÄ± tespit eder
        """
        global layers

        if layer_idx == 0 or layer_idx == len(layers)-1:
            return []  # GiriÅŸ/Ã§Ä±kÄ±ÅŸ katmanlarÄ±nda optimizasyon yapma

        current_layer = layers[layer_idx]
        input_size = len(layers[layer_idx-1])
        output_size = len(layers[layer_idx+1]) if layer_idx+1 < len(layers) else 0

        optimal_sizes = self.calculate_optimal_layer_sizes(input_size, output_size)

        # EÄŸer katman boyutu makul sÄ±nÄ±rlardaysa hiÃ§bir ÅŸey yapma
        if (optimal_sizes['min_hidden'] <= len(current_layer) <= optimal_sizes['max_hidden']):
            return []

        # Fazla nÃ¶ronlarÄ± belirle
        if len(current_layer) > optimal_sizes['max_hidden']:
            # En az etkin nÃ¶ronlarÄ± bul
            neuron_healths = []
            for neuron in current_layer:
                health = self.calculate_neuron_health(neuron)
                neuron_healths.append((health, neuron))

            # SaÄŸlÄ±ÄŸa gÃ¶re sÄ±rala (en dÃ¼ÅŸÃ¼k saÄŸlÄ±klÄ± olanlar Ã¶nce)
            neuron_healths.sort(key=lambda x: x[0])

            # Fazla olan nÃ¶ronlarÄ± seÃ§
            excess_count = len(current_layer) - optimal_sizes['max_hidden']
            excess_neurons = [neuron for (health, neuron) in neuron_healths[:excess_count]]

            return excess_neurons

        return []

    def neuron_level_optimization(self, avg_error):
        """
        GeliÅŸmiÅŸ nÃ¶ron seviyesinde optimizasyon:
        - Fazla nÃ¶ronlarÄ± kaldÄ±rÄ±r
        - Gereksiz nÃ¶ronlarÄ± temizler
        - Eksik nÃ¶ronlarÄ± ekler
        """
        global layers

        # 1. Ã–nce katman boyutlarÄ±nÄ± optimize et
        for layer_idx in range(1, len(layers)-1):  # Hidden layerlar iÃ§in
            excess_neurons = self.detect_excessive_neurons(layer_idx)
            for neuron in excess_neurons:
                self.remove_neuron_from_layer(layer_idx, neuron.id)
                self.log_change('neuron_removed', {
                    'neuron_id': neuron.id,
                    'layer': layer_idx,
                    'reason': 'Excessive neuron count'
                })

        # 2. Sonra normal saÄŸlÄ±k kontrolÃ¼ yap
        health_threshold = max(0.2, min(0.5, 0.3 * (1 + avg_error)))

        for layer_idx, layer in enumerate(layers):
            is_output_layer = (layer_idx == len(layers) - 1)
            is_input_layer = (layer_idx == 0)

            for neuron in layer[:]:  # Kopya Ã¼zerinde dÃ¶ngÃ¼
                health = self.calculate_neuron_health(neuron)

                if health < health_threshold and not is_output_layer and not is_input_layer:
                    if len(self.neuron_health_history.get(neuron.id, [])) >= 3:
                        last_3_health = self.neuron_health_history[neuron.id][-3:]
                        if all(h < health_threshold for h in last_3_health):
                            self.remove_neuron_from_layer(layer_idx, neuron.id)
                            self.log_change('neuron_removed', {
                                'neuron_id': neuron.id,
                                'layer': layer_idx,
                                'health': health,
                                'reason': f'Low health (<{health_threshold:.2f})'
                            })

        # 3. Eksik nÃ¶ronlarÄ± ekle
        for layer_idx in range(1, len(layers)):
            is_output_layer = (layer_idx == len(layers) - 1)

            current_size = len(layers[layer_idx])
            input_size = len(layers[layer_idx-1])
            output_size = len(layers[layer_idx+1]) if layer_idx+1 < len(layers) else 0

            optimal_sizes = self.calculate_optimal_layer_sizes(input_size, output_size)

            required_min = optimal_sizes['min_hidden'] if not is_output_layer else optimal_sizes.get('min_output', 1)

            if current_size < required_min:
                needed = required_min - current_size
                for _ in range(needed):
                    activation = defaultOutActivation if is_output_layer else defaultNeuronActivationType
                    new_neuron = Neuron(activation_type=activation)
                    layers[layer_idx].append(new_neuron)
                    self.log_change('neuron_added', {
                        'neuron_id': new_neuron.id,
                        'layer': layer_idx,
                        'reason': f'Layer too small (added to reach {required_min})'
                    })

        # BaÄŸlantÄ±larÄ± gÃ¼ncelle
        setConnections(preserve_weights=True)



    def layer_level_optimization(self, avg_error):
        """
        KatmanlarÄ± deÄŸerlendirir ve gereksiz olanlarÄ± kaldÄ±rÄ±r
        """
        global layers
        
        if len(layers) <= 2:  # En az giriÅŸ ve Ã§Ä±kÄ±ÅŸ katmanÄ± olmalÄ±
            return
            
        # Gizli katmanlarÄ±n saÄŸlÄ±ÄŸÄ±nÄ± hesapla
        layer_health_scores = []
        for layer_idx in range(1, len(layers)-1):  # Gizli katmanlar
            layer_health = self.calculate_layer_health(layer_idx)
            layer_health_scores.append((layer_idx, layer_health))
            
        # En dÃ¼ÅŸÃ¼k saÄŸlÄ±klÄ± katmanÄ± bul
        layer_health_scores.sort(key=lambda x: x[1])
        worst_layer_idx, worst_health = layer_health_scores[0]
        
        # Katman saÄŸlÄ±k eÅŸiÄŸini hataya gÃ¶re dinamik ayarla
        layer_health_threshold = max(0.3, min(0.6, 0.4 * (1 + avg_error)))
        
        # EÅŸik deÄŸerin altÄ±ndaysa ve yeterli katman varsa sil
        if worst_health < layer_health_threshold and len(layers) > 3:
            self.remove_layer(worst_layer_idx)
            self.log_change('layer_removed', {
                'layer_idx': worst_layer_idx,
                'health': worst_health,
                'reason': f'Low layer health (<{layer_health_threshold:.2f})'
            })
    
    def strategic_growth(self, avg_error):
        """
        AÄŸÄ±n bÃ¼yÃ¼mesini stratejik olarak yÃ¶netir:
        - ZayÄ±f katmanlarÄ± gÃ¼Ã§lendirir
        - Kritik bÃ¶lgelere nÃ¶ron ekler
        - GerektiÄŸinde yeni katman ekler
        """
        # 1. ZayÄ±f katmanlarÄ± gÃ¼Ã§lendir
        self.strengthen_weak_layers(avg_error)
        
        # 2. Kritik bÃ¶lgelere nÃ¶ron ekle
        self.add_neurons_to_critical_areas()
        
        # 3. Gerekirse yeni katman ekle
        self.add_layer_if_needed(avg_error)
    
    def strengthen_weak_layers(self, avg_error):
        """
        ZayÄ±f katmanlara nÃ¶ron ekler
        """
        global layers
        
        complexity_factor = max(0.4, min(0.8, 0.5 * (1 + avg_error)))
        
        for layer_idx in range(1, len(layers)-1):  # Gizli katmanlar
            layer_health = self.calculate_layer_health(layer_idx)
            if layer_health < complexity_factor:
                # Katmana 1-2 nÃ¶ron ekle
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
        YÃ¼ksek hata Ã¼reten veya yÃ¼ksek Ã¶ÄŸrenme potansiyeli olan bÃ¶lgelere nÃ¶ron ekler
        """
        global layers
        
        # En yÃ¼ksek aktivasyon tÃ¼revine sahip nÃ¶ronun katmanÄ±na ekleme yap
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
        AÄŸÄ±n karmaÅŸÄ±klÄ±ÄŸÄ± yeterli deÄŸilse yeni katman ekler
        """
        global layers
        
        if len(layers) >= 5:  # Maksimum 5 katman (giriÅŸ + 3 gizli + Ã§Ä±kÄ±ÅŸ)
            return
            
        # Ortalama katman saÄŸlÄ±ÄŸÄ±nÄ± hesapla
        total_health = 0
        for layer_idx in range(1, len(layers)-1):
            total_health += self.calculate_layer_health(layer_idx)
        avg_health = total_health / (len(layers)-2) if len(layers) > 2 else 0
        
        complexity_factor = max(0.4, min(0.7, 0.5 * (1 + avg_error)))
        
        if avg_health < complexity_factor * 0.7:  # Katmanlar Ã§ok yÃ¼klÃ¼yse
            # En yÃ¼klÃ¼ katmanÄ± bul
            max_load = -1
            busiest_layer = None
            for layer_idx in range(1, len(layers)-1):
                load = len(layers[layer_idx]) * self.calculate_connection_density(layer_idx)
                if load > max_load:
                    max_load = load
                    busiest_layer = layer_idx
                    
            if busiest_layer is not None:
                new_layer_idx = self.insert_hidden_layer(busiest_layer + 1)  # MeÅŸgul katmandan sonra ekle
                self.log_change('layer_added', {
                    'layer_idx': new_layer_idx,
                    'reason': f'High layer load (load={max_load:.2f}, avg_health={avg_health:.2f})'
                })
    
    def calculate_layer_health(self, layer_idx):
        """
        Bir katmanÄ±n genel saÄŸlÄ±k skorunu hesaplar
        """
        global layers
        
        layer = layers[layer_idx]
        if not layer:
            return 0
            
        # Katmandaki nÃ¶ronlarÄ±n ortalama saÄŸlÄ±ÄŸÄ±
        total_health = sum(self.calculate_neuron_health(neuron) for neuron in layer)
        avg_neuron_health = total_health / len(layer)
        
        # KatmanÄ±n baÄŸlantÄ± yoÄŸunluÄŸu
        connection_density = self.calculate_connection_density(layer_idx)
        
        # KatmanÄ±n Ã¶ÄŸrenme potansiyeli (aktivasyon tÃ¼revlerinin ortalamasÄ±)
        learning_potential = np.mean([neuron.activation_derivative() for neuron in layer])
        
        # Katman saÄŸlÄ±k skoru
        layer_health = 0.5 * avg_neuron_health + 0.3 * connection_density + 0.2 * learning_potential
        
        return layer_health
    
    def calculate_connection_density(self, layer_idx):
        """
        Katmandaki baÄŸlantÄ± yoÄŸunluÄŸunu hesaplar
        """
        global layers, connections
        
        if layer_idx == 0:  # GiriÅŸ katmanÄ±
            prev_layer_size = len(layers[layer_idx])
            current_layer_size = len(layers[layer_idx+1])
            total_possible = prev_layer_size * current_layer_size
        elif layer_idx == len(layers)-1:  # Ã‡Ä±kÄ±ÅŸ katmanÄ±
            return 1.0  # Ã‡Ä±kÄ±ÅŸ katmanÄ± iÃ§in maksimum yoÄŸunluk
        else:
            prev_layer_size = len(layers[layer_idx-1])
            current_layer_size = len(layers[layer_idx])
            next_layer_size = len(layers[layer_idx+1])
            total_possible = (prev_layer_size * current_layer_size) + (current_layer_size * next_layer_size)
        
        # GerÃ§ek baÄŸlantÄ± sayÄ±sÄ±nÄ± hesapla
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
        Bir katmanÄ± ve iliÅŸkili baÄŸlantÄ±larÄ± kaldÄ±rÄ±r.
        Silme sonrasÄ± baÄŸlantÄ± sÃ¶zlÃ¼ÄŸÃ¼nÃ¼ tutarlÄ± hale getirmek iÃ§in
        setConnections ile yeniden Ã¶rÃ¼lÃ¼yoruz.
        """
        global layers, connections

        if layer_idx <= 0 or layer_idx >= len(layers)-1:
            print("Hata: GiriÅŸ veya Ã§Ä±kÄ±ÅŸ katmanÄ± silinemez")
            return False

        # KatmanÄ± kaldÄ±r
        del layers[layer_idx]

        # Eski connections anahtarlarÄ±nÄ± da temizle
        # (KeyError vermemesi iÃ§in get ile guardlÄ±yoruz)
        connections.pop(layer_idx-1, None)
        connections.pop(layer_idx,   None)

        # Kalan aÄŸÄ±rlÄ±klarÄ± koruyarak tÃ¼m baÄŸlantÄ±larÄ± yeniden inÅŸa et
        # BÃ¶ylece indekste kayma ya da eksik anahtar kalma riski ortadan kalkar
        setConnections(preserve_weights=True)

        if debug:
            print(f"Katman {layer_idx} silindi ve baÄŸlantÄ±lar yeniden oluÅŸturuldu.")
            

        return True

    
    def insert_hidden_layer(self, position):
        """
        Belirtilen pozisyona yeni bir gizli katman ekler
        """
        global layers, connections

        if position <= 0 or position >= len(layers):
            print("Hata: GeÃ§ersiz katman pozisyonu")
            return False

        # Yeni katman oluÅŸtur (mevcut katmanlarÄ±n ortalamasÄ± kadar nÃ¶ronla)
        size = (len(layers[position - 1]) + len(layers[position])) // 2
        new_layer = [Neuron(activation_type=defaultNeuronActivationType) for _ in range(max(2, size))]

        # KatmanÄ± ekle
        layers.insert(position, new_layer)

        # connections sÃ¶zlÃ¼ÄŸÃ¼ne yeni boÅŸ dict alanÄ± ekle
        connections.insert(position - 1, {})
        connections.insert(position, {})

        # Ã–nceki katmandan yeni katmana baÄŸlantÄ±lar oluÅŸtur
        for prev_neuron in layers[position - 1]:
            for new_neuron in new_layer:
                weight = np.random.uniform(-1, 1) * np.sqrt(2.0 / (len(layers[position - 1]) + len(new_layer)))
                conn = Connection(connectedToArg=[prev_neuron.id, new_neuron.id], weight=weight)

                if prev_neuron.id not in connections[position - 1]:
                    connections[position - 1][prev_neuron.id] = []

                connections[position - 1][prev_neuron.id].append(conn)

        # Yeni katmandan sonraki katmana baÄŸlantÄ±lar oluÅŸtur
        for new_neuron in new_layer:
            for next_neuron in layers[position + 1]:
                weight = np.random.uniform(-1, 1) * np.sqrt(2.0 / (len(new_layer) + len(layers[position + 1])))
                conn = Connection(connectedToArg=[new_neuron.id, next_neuron.id], weight=weight)

                if new_neuron.id not in connections[position]:
                    connections[position][new_neuron.id] = []

                connections[position][new_neuron.id].append(conn)

        return position




# Terminal giriÅŸ dÃ¶ngÃ¼sÃ¼ - Dinamik versiyon

def removeNeuron(layer_index, neuron_index):
    """
    Belirtilen katmandan bir nöron kaldırır ve bağlantıları günceller.
    """
    global layers, connections

    if layer_index < 0 or layer_index >= len(layers):
        print(f"Geçersiz katman indexi: {layer_index}")
        return

    if neuron_index < 0 or neuron_index >= len(layers[layer_index]):
        print(f"Geçersiz nöron indexi: {neuron_index}")
        return

    removed_neuron = layers[layer_index].pop(neuron_index)

    # Tüm bağlantıları güncelle (ağırlıkları korumadan)
    setConnections(preserve_weights=False)

    if debug:
        print(f"Nöron silindi -> Katman: {layer_index}, Nöron ID: {removed_neuron.id}")

def addNeuron(layer_index):
    """
    Belirtilen katmana bir adet yeni nöron ekler ve bağlantıları günceller.
    """
    global layers, connections

    if layer_index < 0 or layer_index >= len(layers):
        print(f"Geçersiz katman indexi: {layer_index}")
        return

    is_output_layer = (layer_index == len(layers) - 1)
    activation = "sigmoid" if is_output_layer else "tanh"

    new_neuron = Neuron(default_value=1, activation_type=activation)
    layers[layer_index].append(new_neuron)

    # Tüm bağlantıları güncelle (ağırlıkları korumadan)
    setConnections(preserve_weights=True)

    if debug:
        print(f"Yeni nöron eklendi -> Katman: {layer_index}, Nöron ID: {new_neuron.id}")


# Command functions

def cmd_refresh(refresh=True,DONTVisualize=False):
    """Refresh the network and visualize"""
    clearGUI()
    runAI()
    visualize_network(layers, connections, refresh=refresh,DONTVisualize=DONTVisualize)
    return getOutput()





def cmd_print_network():
    """Print network structure and connections"""
    output = []
    output.append("=== AÄ YAPISI ===")
    for i, layer in enumerate(layers):
        output.append(f"\nKatman {i} ({len(layer)} nÃ¶ron):")
        for neuron in layer:
            output.append(f"  NÃ¶ron ID: {neuron.id} | DeÄŸer: {neuron.value:.4f} | Aktivasyon: {neuron.activation_type}")
    output.append("\n=== BAÄLANTILAR ===")
    for layer_idx, conn_dict in connections.items():
        output.append(f"\nKatman {layer_idx} -> Katman {layer_idx+1}:")
        for src_id, conn_list in conn_dict.items():
            for conn in conn_list:
                output.append(f"  {src_id} â†’ {conn.connectedTo[1]} | AÄŸÄ±rlÄ±k: {conn.weight:.4f}")
    return "\n".join(output)


def cmd_get_connection(from_id: int, to_id: int) -> str:
    """Get connection weight between two neurons"""
    for layer_idx, conn_dict in connections.items():
        if from_id in conn_dict:
            for conn in conn_dict[from_id]:
                if conn.connectedTo[1] == to_id:
                    return f"BaÄŸlantÄ± bilgisi: {from_id} â†’ {to_id} | AÄŸÄ±rlÄ±k: {conn.weight:.6f}"
    return f"BaÄŸlantÄ± bulunamadÄ±: {from_id} â†’ {to_id}"


def cmd_toggle_visualize() -> str:
    """Toggle network visualization"""
    global visualizeNetwork
    visualizeNetwork = not visualizeNetwork
    return f"GÃ¶rselleÅŸtirme {'aktif' if visualizeNetwork else 'pasif'}"


def cmd_bias(param: str) -> str:
    """Enable or disable or show biases"""
    global bias_is
    if param == "True":
        enable_all_biases()
        bias_is = True
    elif param == "False":
        disable_all_biases()
        bias_is = False
    return f"Bias is : {bias_is}"


def cmd_load_model(filepath: str) -> str:
    """Load model from file"""
    try:
        load_network_optimized(filepath)
        return "Model baÅŸarÄ±yla yÃ¼klendi."
    except Exception:
        traceback.print_exc()
        return "Model yÃ¼klenirken hata oluÅŸtu."


def cmd_train_custom(file_path: str,
                     network_structure=None,
                     epochs=None,
                     batch_size=None,
                     learning_rate=None,
                     intelligenceValue=None,useDynamicModelChanges=True,symbol="",epochNumberForLimitError=None) -> str:
    """Train network with custom data"""
    try:
        setLayers(network_structure or [2,4,1])
        X, y = modeltrainingprogram.read_csv_file(file_path)
        train_kwargs = {}
        if epochs is not None:
            train_kwargs['epochs'] = epochs
        if batch_size is not None:
            train_kwargs['batch_size'] = batch_size
        if learning_rate is not None:
            train_kwargs['learning_rate'] = learning_rate
        if intelligenceValue is not None:
            train_kwargs['intelligenceValue'] = intelligenceValue
        train_network(X, y, **train_kwargs,useDynamicModelChanges=useDynamicModelChanges,symbol=symbol,epochNumberForLimitError=epochNumberForLimitError)
        hata=testModel(file_path.replace("trainingDatas/", "trainingDatas/test"),inputNum=network_structure[0],targetNum=network_structure[-1],DONTVisualize=True)
        return "EÄŸitim tamamlandÄ±. Hata:"+str(hata)
    except Exception as e:
        traceback.print_exc()
        return f"Hata: {e}"


def cmd_change_weight(from_id: int, to_id: int, new_weight: float) -> str:
    """Change connection weight"""
    try:
        change_weight(connections, from_id, to_id, new_weight)
        return f"BaÄŸlantÄ± aÄŸÄ±rlÄ±ÄŸÄ± gÃ¼ncellendi: {from_id} â†’ {to_id} = {new_weight:.4f}"
    except Exception as e:
        traceback.print_exc()
        return f"Hata: {e}"


def cmd_change_neuron(id: int, new_value: float) -> str:
    """Change neuron value"""
    try:
        get_neuron_by_id(id).value = new_value
        return f"NÃ¶ron {id} deÄŸeri gÃ¼ncellendi: {new_value:.4f}"
    except Exception as e:
        traceback.print_exc()
        return f"Hata: {e}"


def cmd_set_input(values: list) -> dict:
    """Set input layer values and return stats"""
    layer0 = layers[0]
    stats = {}
    try:
        # assign values
        for i, val in enumerate(values[:len(layer0)]):
            layer0[i].value = val
        stats['min'] = float(np.min(values[:len(layer0)]))
        stats['max'] = float(np.max(values[:len(layer0)]))
        stats['mean'] = float(np.mean(values[:len(layer0)]))
        stats['values'] = values[:len(layer0)]
        return stats
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}

def denormalize_value(norm_val, min_val, max_val):
    return ((norm_val + 1) / 3) * (max_val - min_val) + min_val

def denormalize_value0_1(norm_x, min_val, max_val):
    """
    0-1 aralÄ±ÄŸÄ±ndaki bir deÄŸeri min_val ile max_val aralÄ±ÄŸÄ±na geri dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r.
    """
    return norm_x * (max_val - min_val) + min_val





def cmd_help():
    print("\nKullanÄ±labilir Komutlar:")
    print("- refresh(): AÄŸÄ± yenile")
    print("- print_network(): AÄŸ yapÄ±sÄ±nÄ± terminalde gÃ¶ster")
    print("- get_connection(from_id,to_id): BaÄŸlantÄ± aÄŸÄ±rlÄ±ÄŸÄ±nÄ± gÃ¶ster")
    #print("- draw(): Ã‡izim modunu aÃ§")
    print("- addNeuron(layer_idx): Yeni nöron ekle !!!BUNU KULLANMA eğer bilgisayar değilsen!!!")
    print("- removeNeuron(layer_idx,neuron_id): Nöron sil !!!BUNU KULLANMA eğer bilgisayar değilsen!!!")
        
    #print("- addLayer(index;[nÃ¶ronlar]): Katman ekle")
    #print("- removeLayer(index): Katman sil")
    print("- changeW(from;to;weight): AÄŸÄ±rlÄ±k deÄŸiÅŸtir")
    print("- changeN(id;value): NÃ¶ron deÄŸeri deÄŸiÅŸtir")
    print("- visualize(): AÄŸ gÃ¶rselleÅŸtirmeyi aÃ§/kapat")
    #print("- train_mnist([epochs;batch;lr;test_size;intel]): MNIST ile eÄŸitim")
    #print("- train_digits([epochs;batch;lr;test_size;intel]): Digits ile eÄŸitim")
    print("- train_custom(dosya.csv[;epochs;batch;lr;test_size;intel]): Ã–zel veri ile eÄŸitim")
    print("- set_input values:giriÅŸ deÄŸerlerini belirle")
    print("- bias(True or False) :bias deÄŸerlerini aÃ§Ä±p kapatmaya yarÄ±yor boÅŸ bÄ±rakÄ±lÄ±rsa aÃ§Ä±k mÄ± kapalÄ± mÄ± onu veriyor")
    print("- history() :geÃ§miÅŸte yazÄ±lan komutlarÄ± veriyor")
    print("- loadModel(dosyaAdÄ±) : hazÄ±r modeli dosyadan yÃ¼klÃ¼yor")
    print("-----------Crypto Commands-----------")
  
    
    print("- exit: Programdan Ã§Ä±k")





import numpy as np
import sounddevice as sd
import librosa
from queue import Queue
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LiveLogMelExtractor:
    def __init__(self, visualize=False):
        # Ayarlar
        self.sample_rate = 16000
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.n_mels = 40
        self.update_interval = 100  # ms
        
        # Sistem değişkenleri
        self.visualize = visualize
        self.audio_queue = Queue()
        self.is_recording = False
        self.stream = None
        self.audio_buffer = np.array([])
        self.current_logmel = None
        
        # Görselleştirme için
        self.fig = None
        self.ax = None
        self.im = None
        self.ani = None

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Ses akışı hatası: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def compute_logmel(self, audio):
        if len(audio) < self.win_length:
            return None
            
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            window='hamming',
            power=2.0
        )
        return librosa.power_to_db(S)

    def process_audio(self):
        while not self.audio_queue.empty():
            audio_chunk = self.audio_queue.get()
            self.audio_buffer = np.append(self.audio_buffer, audio_chunk.flatten())
            
            if len(self.audio_buffer) >= self.win_length:
                self.current_logmel = self.compute_logmel(self.audio_buffer[-self.win_length:])
                
                if self.current_logmel is not None:
                    # Görselleştirme aktifse güncelle
                    if self.visualize and self.im is not None:
                        self.im.set_array(self.current_logmel)
                        self.im.set_extent([0, self.current_logmel.shape[1], 0, self.n_mels])
                        self.im.autoscale()
                    
                    # Logmel verilerini konsola yazdır
                    self.print_logmel_info()
    
    def print_logmel_info(self):
        """Logmel verilerini konsola yazdır"""
        if self.current_logmel is not None:
            print("\n" + "="*50)
            print(f"LogMel Spektrogram Verisi (Boyut: {self.current_logmel.shape})")
            print(f"Min Değer: {np.min(self.current_logmel):.2f} dB")
            print(f"Max Değer: {np.max(self.current_logmel):.2f} dB")
            print(f"Ortalama: {np.mean(self.current_logmel):.2f} dB")
            print("Son 5 zaman noktasının ortalaması:")
            print(np.mean(self.current_logmel[:, -5:], axis=1))
            print("Bütün logmel verisinin boyutu:",len(self.current_logmel))
            print("="*50 + "\n")

    def get_current_logmel(self):
        """Güncel logmel verisini döndür"""
        return self.current_logmel.copy() if self.current_logmel is not None else None

    def update_plot(self, frame):
        self.process_audio()
        return [self.im] if self.im is not None else []

    def start(self):
        print("Canlı Log-Mel Spektrogram İşleyici")
        print(f"Görselleştirme: {'Açık' if self.visualize else 'Kapalı'}")
        print("Kayıt başlatılıyor... (Çıkmak için Ctrl+C)")
        
        self.is_recording = True
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.hop_length
        )
        self.stream.start()
        
        if self.visualize:
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
            self.ani = FuncAnimation(
                self.fig,
                self.update_plot,
                interval=self.update_interval,
                blit=True,
                cache_frame_data=False
            )
            plt.tight_layout()
            plt.show()
        else:
            try:
                while self.is_recording:
                    self.process_audio()
                    time.sleep(self.update_interval/1000)
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Kayıt durduruldu.")

if __name__ == "__main__" and False:
    # Örnek kullanım:
    # Görselleştirme ile çalıştırma:
    # extractor = LiveLogMelExtractor(visualize=True)
    
    # Görselleştirme olmadan çalıştırma:
    extractor = LiveLogMelExtractor(visualize=False)
    
    try:
        extractor.start()
    except KeyboardInterrupt:
        extractor.stop()
    except Exception as e:
        extractor.stop()
        raise e

cmd_toggle_visualize()
cmd_train_custom(file_path="parity_problem.csv",network_structure=[4,8,2],epochs=0.1,learning_rate=1)


cmd_set_input([0,1,0,0])

cmd_refresh(refresh=False)


removeNeuron(0,2)

cmd_refresh(refresh=False)

addNeuron(0)

cmd_refresh(refresh=False)


cmd_train_custom(file_path="parity_problem.csv",network_structure=[4,8,2],epochs=0.1,learning_rate=1)
cmd_set_input([0,1,0,0])

cmd_refresh(refresh=False)