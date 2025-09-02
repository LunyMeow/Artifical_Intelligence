import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
import math

# --- Veri yükleme ---
values = []
invalid_indexes = []

with open("network_changes.log", "r") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line.split(" ")[0]))
        except ValueError:
            values.append(None)
            if line == "Stopped due targetError...":
                invalid_indexes.append([i,0])
            elif line == "Stopped due closing to zero slope...":
                invalid_indexes.append([i,2])
            elif line.startswith("lr changed "):
                invalid_indexes.append([i,1])
            elif line.startswith("Spesificed Neuron removed from layer"):
                invalid_indexes.append([i,3])
            elif line.startswith("Neuron added to hidden layer"):
                invalid_indexes.append([i,4])
            elif line.startswith("Neuron removed from layer"):
                invalid_indexes.append([i,5])
            elif line == "Model rebooted":
                invalid_indexes.append([i,6])
            else:
                invalid_indexes.append([i,-1])

x = np.arange(1, len(values) + 1)
y_valid = np.array([v if v is not None else np.nan for v in values])

# --- Grafik çizimi ---
fig, ax = plt.subplots()
line_plot, = ax.plot(x, y_valid, marker="o", linestyle="-", color="b", label="Geçerli Veriler")

color_label_map = {
    0: ("green", "Reached TargetError"),
    1: ("blue", "Lr changed"),
    2: ("green", "Slope close to zero"),
    3: ("orange", "Specific neuron removed"),
    4: ("orange", "Neuron added"),
    5: ("brown", "Neuron removed"),
    6:("red","Model rebooted"),
    -1: ("black", "Other")
}

used_labels = set()
for idx, errorNumber in invalid_indexes:
    color, label = color_label_map.get(errorNumber, ("black", "Unknown"))
    if label in used_labels or errorNumber == -1:
        plot_label = ""
    else:
        plot_label = label
        used_labels.add(label)

    ax.scatter(idx - 1, 0, color=color, s=25, zorder=5, label=plot_label)



    

ax.set_title("Veri Grafiği")
ax.set_xlabel("Index")
ax.set_ylabel("Değer")
ax.grid(True)
ax.legend()

# --- Eğim hesaplama fonksiyonu (C++ mantığı) ---
def compute_slope(y_vals, n_recent=10):
    N = len(y_vals)
    if N < 2:
        return 0.0, 0.0

    # Son n veri için basit eğim
    slope_recent = 0.0
    if N >= n_recent + 1:
        slope_recent = (y_vals[-1] - y_vals[-n_recent-1]) / n_recent

    # Tüm veriler için lineer regresyon
    x_indices = np.arange(N)
    mask = ~np.isnan(y_vals)
    x_masked = x_indices[mask]
    y_masked = y_vals[mask]

    if len(x_masked) < 2:
        slope_overall = 0.0
    else:
        sumX = np.sum(x_masked)
        sumY = np.sum(y_masked)
        sumXY = np.sum(x_masked * y_masked)
        sumXX = np.sum(x_masked * x_masked)
        slope_overall = (len(x_masked) * sumXY - sumX * sumY) / (len(x_masked) * sumXX - sumX * sumX)

    return slope_recent, slope_overall

# --- SpanSelector ile seçilen aralığın eğimini hesaplama ---
text_handle = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                      fontsize=12, verticalalignment='top',
                      bbox=dict(facecolor='white', alpha=0.5))

def onselect(xmin, xmax):
    start_idx = max(int(np.floor(xmin)) - 1, 0)
    end_idx = min(int(np.ceil(xmax)), len(x))

    y_range = y_valid[start_idx:end_idx]

    slope_recent, slope_overall = compute_slope(y_range, n_recent=10)

    # X ve Y’yi normalize ederek açı hesapla
    x_range = np.arange(len(y_range))
    mask = ~np.isnan(y_range)
    x_norm = x_range[mask]
    y_norm = y_range[mask]
    if len(x_norm) >= 2:
        dx = x_norm[-1] - x_norm[0]
        dy = y_norm[-1] - y_norm[0]
        if dx == 0:
            angle = 90.0
        else:
            angle = math.degrees(math.atan2(dy, dx))
    else:
        angle = 0.0

    print(f"Son 10 veri eğimi: {slope_recent}")
    print(f"Seçilen aralık genel eğim (lineer regresyon): {slope_overall}")
    print(f"Seçilen aralığın açısı: {angle:.2f}°")
    text_handle.set_text(f"Son 10 veri eğimi: {slope_recent:.4f}\n"
                         f"Genel eğim: {slope_overall:.4f}\n"
                         f"Açı: {angle:.2f}°")

span = SpanSelector(ax, onselect, 'horizontal', useblit=False,
                    props=dict(alpha=0.5, facecolor='yellow'))

plt.show()
