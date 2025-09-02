
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)

layers = [2, 8, 1]

class XORNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                x = self.tanh(x)
        x = self.sigmoid(x)
        return x

model = XORNet(layers).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 4000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")



def visualize_model(model, layers):
    """
    model: PyTorch modeli
    layers: katman boyutları listesi, örn [2,8,1]
    """
    # Global nöron ID oluştur
    global_ids = []
    counter = 0
    for sz in layers:
        ids = list(range(counter, counter+sz))
        global_ids.append(ids)
        counter += sz

    plt.figure(figsize=(10,6))
    
    # Bağlantıları çiz
    for l_idx, layer in enumerate(model.layers):
        weight = layer.weight.detach()
        max_w = weight.abs().max().item()  # maksimum ağırlık
        for out_idx in range(weight.shape[0]):
            for in_idx in range(weight.shape[1]):
                start = (l_idx, global_ids[l_idx][in_idx])
                end = (l_idx+1, global_ids[l_idx+1][out_idx])
                w = weight[out_idx, in_idx].item()
                lw = max(abs(w)/max_w*5, 0.5)  # normalize kalınlık
                plt.plot([start[0], end[0]], [start[1], end[1]],
                         'b', alpha=min(max(abs(w),0.1),1.0), linewidth=lw/2)

    # Nöronları çiz ve ID yazdır
    for l_idx, layer_ids in enumerate(global_ids):
        for nid_idx, nid in enumerate(layer_ids):
            plt.scatter(l_idx, nid, s=300, color='r')
            # Katman içi ID ve global ID birlikte
            plt.text(l_idx, nid, f"{nid_idx}({nid})", color='w', ha='center', va='center')

    plt.xlabel("Katmanlar")
    plt.ylabel("Nöronlar (Global ID)")
    plt.title("XORNet Nöronları ve Bağlantıları")
    plt.show()

# Eğitim sonrası görselleştir





# Eğitim sonrası
with torch.no_grad():
    pred = model(X)
    print("\nHam Çıkışlar:")
    print(pred)
    print("\nTahminler (0 veya 1):")
    print(torch.round(pred))

    print("\n--- Katman Bağlantıları ---")
    for i, layer in enumerate(model.layers):
        weight = layer.weight
        bias = layer.bias
        print(f"\nKatman {i+1}:")
        for out_idx in range(weight.shape[0]):        # her çıkış nöronu
            for in_idx in range(weight.shape[1]):     # her giriş nöronu
                print(f"n{in_idx} -> n{out_idx} = {weight[out_idx, in_idx].item():.4f}")
            print(f"Bias of n{out_idx} = {bias[out_idx].item():.4f}")
    
# Eğitim sonrası görselleştir
    visualize_model(model, layers)
