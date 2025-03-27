import csv
import traceback
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
import math
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


visualizeNetwork =True





class Neuron:
    def __init__(self, default_value=0.0, id=0, activation_type='sigmoid'):
        self.value = default_value
        self.id = id
        self.activation_type = activation_type
        self.output = 0.0  # Çıktı değeri, aktivasyon fonksiyonundan sonra hesaplanacak
        self.weightedSum=0
    def activation(self, x):
        if self.activation_type == 'sigmoid':
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
        # Her katmandan gelen bağlantıları kontrol et
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



# Ağ oluşturma


layers = [[Neuron(0,0),Neuron(0,1),Neuron(0,2),Neuron(0,3),Neuron(0,4),Neuron(0,5),Neuron(0,6),Neuron(0,7),Neuron(0,8)],
          [Neuron(0,9),Neuron(0,10),Neuron(0,11),Neuron(0,12),Neuron(0,13),Neuron(0,14),Neuron(0,15),Neuron(0,16),Neuron(0,17),Neuron(0,18),Neuron(0,19)],
          [Neuron(0,20),Neuron(0,21),Neuron(0,22),Neuron(0,23),Neuron(0,24),Neuron(0,25),Neuron(0,26),Neuron(0,27),Neuron(0,28),Neuron(0,29)],
          [Neuron(0,30)]]

# Bağlantıları oluşturma
connections = {layer_idx: {} for layer_idx in range(len(layers) - 1)}

# Bağlantıları oluştururken rastgele ağırlıklar atayın
for layer_idx in range(len(layers) - 1):
    for neuron in layers[layer_idx]:
        for next_neuron in layers[layer_idx + 1]:
            conn = Connection(fromTo=[neuron.id, next_neuron.id], weight=random.uniform(-1.0, 1.0))  # Rastgele ağırlık
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

fig, ax = plt.subplots(figsize=(7,7)) 
def RandomInput(event):
    for i in layers[0]:
        i.value = random.uniform(0.1, 1)
    for layer in layers[1:]:
        for neuron in layer:
            neuron.calculate_weighted_sum(layers, connections)
    visualize_network(layers, connections,fig,ax)  # Güncellenmiş ağı göster
# Butonu yalnızca bir kez oluştur
#ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])  # [sol, alt, genişlik, yükseklik]




def randomWeights(event,connections=connections, min_weight=-1.0, max_weight=1.0):
    """
    Tüm bağlantıların ağırlıklarını rastgele günceller.
    
    :param connections: Katmanlar arası bağlantılar (dict)
    :param min_weight: Minimum ağırlık değeri
    :param max_weight: Maksimum ağırlık değeri
    """
    for layer in connections.values():
        for neuron_id, conn_list in layer.items():
            for conn in conn_list:
                conn.weight = random.uniform(min_weight, max_weight)
                print(f"Güncellendi: {conn.connectedTo[0]} -> {conn.connectedTo[1]}, Yeni Weight: {conn.weight}")

    visualize_network(layers, connections,fig,ax)  # Güncellenmiş ağı göster
def on_mouse_click(event):
    if event.button == 1:  # Sol tıklama
        RandomInput(event)
    elif event.button == 3:  # Sağ tıklama
        randomWeights(event)

fig.canvas.mpl_connect('button_press_event', on_mouse_click)


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

# Örnek kullanım
x = 0.5  # 0 ile 1 arasında bir değer
scaled_value = scale_value(x, 0, 1, 0, 8)
print(scaled_value)  # Çıktı: 4.0
def runAI():
    for layer in layers[1:]:
        for neuron in layer:
            #print(f"Nöron {neuron.id}: {neuron.value}")
            neuron.calculate_weighted_sum(layers,connections)
    print(f"Son değer: {scale_value(get_neuron_by_id(30).value,0,1,0,8)}")




def visualize_network(layers, connections, figp=fig, axp=ax):
    if not visualizeNetwork:
        return None
    runAI()
    G = nx.DiGraph()
    pos = {}
    edge_labels = {}
    node_labels = {}
    edge_weights = []  # Ağırlık listesi

    

    # Layer bazlı konumlandırma
    for layer_idx, layer in enumerate(layers):
        for neuron_idx, neuron in enumerate(layer):
            G.add_node(neuron.id, value=neuron.value , weightedSum = neuron.weightedSum)
            pos[neuron.id] = (layer_idx, -neuron_idx)
            node_labels[neuron.id] = f"{neuron.id}"

    # Bağlantıları ekleme
    for layer_idx, layer_connections in connections.items():
        for neuron_id, conn_list in layer_connections.items():
            for conn in conn_list:
                G.add_edge(conn.connectedTo[0], conn.connectedTo[1], weight=conn.weight)
                edge_labels[(conn.connectedTo[0], conn.connectedTo[1])] = f"{conn.weight:.2f}"
                edge_weights.append(conn.weight)

    """
    cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'berlin', 'managua', 'vanimo']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])]"""
    # Ağırlıkları normalize et (-1 ile 1 arasında olduğu varsayılıyor)
    if edge_weights:
        min_w, max_w = min(edge_weights), max(edge_weights)
        norm_weights = [(w - min_w) / (max_w - min_w) if max_w != min_w else 0.5 for w in edge_weights]
        edge_colors = [plt.cm.Greens(w) for w in norm_weights]  # Büyük ağırlıklar siyah, küçük ağırlıklar beyaz
    
    # Nöron değerlerini normalize et ve renkleri ayarla
    values = [neuron.value for layer in layers for neuron in layer]
    norm = plt.Normalize(min(values), max(values))
    colors = plt.cm.Blues(norm(values))  # Mavi tonları

    axp.clear()

    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=colors, edge_color=edge_colors, width=2, arrows=True)

    # Fare hareketlerini izlemek için olay bağlama
    def on_hover(event):
        if event.inaxes:
            for neuron_id, (x, y) in pos.items():
                if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:
                    neuron = get_neuron_by_id(neuron_id)
                    if neuron:
                        axp.set_title(f"Nöron ID: {neuron.id}, Değer: {neuron.value:.2f} , Ağırlıklı Ort: {neuron.weightedSum}")
                        figp.canvas.draw()

    figp.canvas.mpl_connect("motion_notify_event", on_hover)

    # Yük etiketlerini uygun bir şekilde konumlandırma
    for (start, end), weight in edge_labels.items():
        x_start, y_start = pos[start]
        x_end, y_end = pos[end]
        x_label, y_label = isaret_koy((x_start, y_start), (x_end, y_end), 0.2)
        plt.text(x_label, y_label, weight, color='red', ha='center', va='center', fontsize=10)

    axp.set_facecolor("darkgray")  # Ekseni siyah yap
    figp.patch.set_facecolor("darkgray")  # Figürün arka planını siyah yap
    plt.axis('off')  # Eksenleri kapat
    
    plt.draw()  # Figürü güncelle
    plt.pause(0.01)  # Güncellemeleri görmek için kısa bir süre bekle





def isaret_koy(A, B, mesafe_orani):
    # A ve B noktalarının koordinatlarını al
    x1, y1 = A
    x2, y2 = B

    # Ok boyunca belirtilen mesafeye göre yeni noktanın koordinatlarını hesapla
    x_isaret = x1 + (x2 - x1) * mesafe_orani
    y_isaret = y1 + (y2 - y1) * mesafe_orani

    return x_isaret, y_isaret


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



def add_neuron(layers, connections, layer_idx, new_neuron):
    """
    Yeni bir nöron ekler ve bu nöronu önceki ve sonraki katmandaki nöronlarla bağlar.

    :param layers: Nöron katmanları
    :param connections: Katmanlar arası bağlantılar
    :param layer_idx: Yeni nöronun ekleneceği katman indeksi
    :param new_neuron: Yeni eklenecek nöron (Neuron objesi)
    """
    # Yeni nöronu ilgili katmana ekle
    layers[layer_idx].append(new_neuron)

    # Yeni nöronu, önceki katmandaki nöronlarla bağla
    if layer_idx > 0:  # Katman var mı diye kontrol et
        for prev_neuron in layers[layer_idx - 1]:
            conn = Connection(fromTo=[prev_neuron.id, new_neuron.id], weight=random.uniform(-1.0, 1.0))
            if prev_neuron.id not in connections[layer_idx - 1]:
                connections[layer_idx - 1][prev_neuron.id] = []
            connections[layer_idx - 1][prev_neuron.id].append(conn)

    # Yeni nöronu, sonraki katmandaki nöronlarla bağla
    if layer_idx < len(layers) - 1:  # Sonraki katman var mı diye kontrol et
        for next_neuron in layers[layer_idx + 1]:
            conn = Connection(fromTo=[new_neuron.id, next_neuron.id], weight=random.uniform(-1.0, 1.0))
            if new_neuron.id not in connections[layer_idx]:
                connections[layer_idx][new_neuron.id] = []
            connections[layer_idx][new_neuron.id].append(conn)

    print(f"Nöron ID {new_neuron.id} katman {layer_idx} eklendi ve bağlantılar kuruldu.")
    visualize_network(layers,connections)





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








def normalize_weights(connections):
    all_weights = [conn.weight for layer_connections in connections.values() for conn_list in layer_connections.values() for conn in conn_list]
    min_weight = min(all_weights)
    max_weight = max(all_weights)
    
    for layer_connections in connections.values():
        for neuron_id, conn_list in layer_connections.items():
            for conn in conn_list:
                conn.weight = 2 * ((conn.weight - min_weight) / (max_weight - min_weight)) - 1



def TrainFor(inputValues, targetValues, connections, learning_rate=0.1):
    """
    Sinir ağını eğitmek için ileri yayılım, hata hesaplama, geri yayılım ve ağırlık güncelleme işlemlerini gerçekleştirir.

    :param inputValues: Giriş verisi (tahta durumu, örneğin [0, 0, 0, 0, 0, 0, 1, 0, 0])
    :param targetValues: Hedef veri (one-hot vektörü, örneğin [1, 0, 0, 0, 0, 0, 0, 0, 0])
    :param connections: Ağırlık bağlantıları
    :param learning_rate: Öğrenme oranı
    """
    # İleri yayılım (Forward Propagation)
    for i, value in enumerate(inputValues):
        layers[0][i].value = value  # Giriş katmanındaki nöronlara değerleri ata

    runAI()

    # Hata hesaplama
    output_layer = layers[-1]
    errors = []
    for i, neuron in enumerate(output_layer):
        error = targetValues[i] - neuron.value
        errors.append(error)

    # Geri yayılım (Backward Propagation)
    deltas = {}
    for layer_idx in range(len(layers) - 1, 0, -1):  # Çıkış katmanından giriş katmanına doğru
        layer = layers[layer_idx]
        for neuron_idx, neuron in enumerate(layer):
            if layer_idx == len(layers) - 1:  # Çıkış katmanı
                error = errors[neuron_idx]
                delta = error * neuron.activation_derivative()
            else:  # Gizli katmanlar
                delta = 0
                for next_neuron in layers[layer_idx + 1]:
                    for conn in connections[layer_idx].get(neuron.id, []):
                        if conn.connectedTo[1] == next_neuron.id:
                            delta += conn.weight * deltas[next_neuron.id]
                delta *= neuron.activation_derivative()
            deltas[neuron.id] = delta

    # Ağırlık güncelleme
    for layer_idx in range(len(layers) - 1):
        for neuron in layers[layer_idx]:
            for conn in connections[layer_idx].get(neuron.id, []):
                to_neuron = get_neuron_by_id(conn.connectedTo[1])
                conn.weight += learning_rate * deltas[to_neuron.id] * neuron.value

    # Hata payını yazdır
    print(f"Hata Payı: {hata_payi(targetValues, [neuron.value for neuron in layers[-1]])}")

def load_training_data(file_path):
    training_data = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Başlık satırını atla
        for row in reader:
            state = list(map(int, row[:9]))  # İlk 9 değer (H1-H9)
            move = int(row[9])              # Hamle (Move)
            reward = int(row[10])           # Ödül (Reward)
            training_data.append((state, move, reward))
    return training_data

def prepare_training_data(training_data):
    input_data = []
    target_data = []
    
    for state, move, reward in training_data:
        input_data.append(state)  # Giriş verisi (tahta durumu)
        
        # Hedef veri (one-hot vektörü)
        target = [0] * 9
        target[move] = 1  # Hamle indeksini 1 yap
        target_data.append(target)
    
    return input_data, target_data

def train_network(input_data, target_data, connections, learning_rate=0.1, epochs=1000):
    for epoch in range(epochs):
        total_error = 0  # Toplam hata payı
        for inputValues, targetValues in zip(input_data, target_data):
            # İleri yayılım ve geri yayılım
            TrainFor(inputValues, targetValues, connections, learning_rate)
            
            # Hata payını hesapla ve topla
            output_values = [neuron.value for neuron in layers[-1]]
            total_error += hata_payi(targetValues, output_values)
        
        # Her epoch'ta ortalama hata payını yazdır
        print(f"Epoch {epoch + 1}/{epochs}, Ortalama Hata Payı: {total_error / len(input_data)}")





# Terminal giriş döngüsü
cmd = "runAI()"
while True:
    visualize_network(layers,connections)
    #visualize_network(layers,connections)
    if cmd == "exit":
        break
    if cmd == "continue()":
        visualize_network(layers,connections)
    elif cmd.startswith("add_neuron("):
        try:
            # Yeni nöron eklemek için komutla alınan argümanları işlemeye çalışıyoruz
            args = cmd[11:-1].split(",")
            layer_idx = int(args[0])
            new_neuron_value = float(args[1])

            total_neurons = sum(len(layer) for layer in layers)
            #new_neuron_id = int(args[2] if args[2]!=None else total_neurons)
            
            new_neuron = Neuron(default_value=new_neuron_value, id=total_neurons+1)
            add_neuron(layers, connections, layer_idx, new_neuron)
            
            
        except Exception as error:
            print("Hatalı giriş! Örnek format: add_neuron(layerId,newNeuronValue,neuronId=nextId(default))")
    elif (cmd.startswith("changeW(") and cmd.endswith(")")): 
        try:
            args = cmd[8:-1].split(",")
            from_id, to_id, new_weight = int(args[0]), int(args[1]), float(args[2])
            change_weight(connections, from_id, to_id, new_weight)

        except Exception as error:
            print("Hatalı giriş! Örnek format: changeW(0,5,0.5)")
            traceback.print_exc()
    elif (cmd.startswith("changeN(") and cmd.endswith(")")): 
        try:
            args = cmd[8:-1].split(",")
            id, newValue = int(args[0]),float(args[1])
            get_neuron_by_id(id).value=newValue

        except Exception as error:
            print("Hatalı giriş! Örnek format: changeN(0,0.1)")
            print(error)
    elif (cmd.startswith("getNeuronV(")and cmd.endswith(")")):
        args = cmd[11:-1].split(",")
        for i in args:
            print(get_neuron_by_id(int(i)).calculate_weighted_sum(layers,connections))
            
    elif (cmd.startswith("runAI(")and cmd.endswith(")")):
        runAI()
                
    elif (cmd.startswith("changeInputRandomly(")and cmd.endswith(")")):
        for i in layers[1]:
            i.value=random.uniform(0.1, 1)
    elif (cmd.startswith("trainFor(")and cmd.endswith(")")):
        # Komutu parçalara ayıralım
        #parts = cmd.split('(')[1].split(')')[0]  # Parantez içeriğini alıyoruz
        #values = parts.split('],')  # Her bir parametreyi ayırıyoruz

        # Tüm parametreleri işliyoruz
        #values = [val.replace('[', '').replace(']', '') for val in values]  # Parantezleri kaldırıyoruz
        #values = [list(map(float, val.split(','))) for val in values]  # Virgülle ayırıp float'a çeviriyoruz

        try:
            # Verileri yükle
            training_data = load_training_data("xox_training_data.csv")

            # Verileri eğitim formatına dönüştür
            input_data, target_data = prepare_training_data(training_data)

            # Eğitimi başlat
            train_network(input_data, target_data, connections, learning_rate=0.1, epochs=100)
        except Exception as errorThunderStorm:
            traceback.print_exc()
            print("Yanlış kod dizimi veya hata örnek kod:\ntrainFor(input_values, output_values, target_values)")
    elif (cmd.startswith("setInput(")and cmd.endswith(")")):
        args = cmd[9:-1].split(",")
        for neuron,value in enumerate(args):
            get_neuron_by_id(neuron).value=float(value)
        pass
    else:
        print("Geçersiz komut! Örnek: add_neuron(1,10,0.5) veya exit")
    
    visualize_network(layers, connections,fig,ax)  # Güncellenmiş ağı göster

    cmd = input("Komut girin (add_neuron(layer_idx,new_neuron_id,value) veya exit): ")
