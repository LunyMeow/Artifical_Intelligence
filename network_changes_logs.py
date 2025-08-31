import matplotlib.pyplot as plt

values = []
invalid_indexes = []  # Geçersiz verilerin konumu

with open("network_changes.log", "r") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            # Sayıya çevrilemeyen veri varsa None ekle
            values.append(None)
            if line == "Stopped due targetError...":
                invalid_indexes.append([i,0])
            elif line == "Stopped due closing to zero slope...":
                invalid_indexes.append([i,1])



x = list(range(1, len(values) + 1))

# Normal değerleri çiz
y_valid = [v if v is not None else float("nan") for v in values]
plt.plot(x, y_valid, marker="o", linestyle="-", color="b", label="Geçerli Veriler")

for i, (idx, errorNumber) in enumerate(invalid_indexes):
    if errorNumber == 0:
        plt.scatter(idx, 0, color="red", s=25, zorder=5,
                    label="Reached TargetError" if i == 0 else "")
    elif errorNumber == 1:
        plt.scatter(idx, 0, color="green", s=25, zorder=5,
                    label="Zero slope" if i == 0 else "")

plt.title("Veri Grafiği")
plt.xlabel("Index")
plt.ylabel("Değer")
plt.grid(True)
plt.legend()
plt.show()
