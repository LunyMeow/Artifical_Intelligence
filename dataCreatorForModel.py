import yfinance as yf
import csv

def write_to_csv(filename, dataa, input_count, target_count):
    """
    data: [[input1, input2, ..., target1, target2], ...]
    input_count: kaç tane input değeri var
    """
    header = [f"input{i+1}" for i in range(input_count)] + ["target1", "target2"]
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        print(dataa)
        writer.writerows(dataa)

    print(f"CSV dosyasına yazıldı: {filename}")

csvData = []
all_data = []  # Tüm veriyi tutacağız
for i in range(1, 20):
    startDate = f"2023-01-{i:02d}"
    endDate = f"2023-02-{i:02d}"

    data = yf.download("BTC-USD", start=startDate, end=endDate, interval="1d")
    
    # Tüm verileri topla
    for index, row in data.iterrows():
        close_value = float(row['Close'])
        all_data.append(close_value)

# Tüm verinin min ve max değerlerini bul
min_value = min(all_data)
max_value = max(all_data)

# Verileri normalize et
for i in range(1, 20):
    startDate = f"2023-01-{i:02d}"
    endDate = f"2023-02-{i:02d}"

    data = yf.download("BTC-USD", start=startDate, end=endDate, interval="1d")
    inputData = []
    for index, row in data.iterrows():
        close_value = float(row['Close'])
        normalized_value = (close_value - min_value) / (max_value - min_value)  # Normalizasyon
        inputData.append(round(normalized_value,3))
        print(f"Tarih: {index.date()} | Normalized Close: {normalized_value}")

    beforeDataValue = yf.download("BTC-USD", start=f"2023-02-{i:02d}", end=f"2023-02-{i+1:02d}", interval="1d")
    if not beforeDataValue.empty:
        beforeTargetValue = beforeDataValue['Close'].iloc[0].item()
        beforeNormalized = (beforeTargetValue - min_value) / (max_value - min_value)  # Normalizasyon
        print(f"Before Normalized Close: {beforeNormalized}")
    else:
        print("Veri bulunamadı.")

    targetDataValue = yf.download("BTC-USD", start=f"2023-02-{i+1:02d}", end=f"2023-02-{i+2:02d}", interval="1d")

    if not targetDataValue.empty:
        targetCloseValue = targetDataValue['Close'].iloc[0].item()
        targetNormalized = (targetCloseValue - min_value) / (max_value - min_value)  # Normalizasyon
        print(f"Target Normalized Close: {targetNormalized}")
    else:
        print("Veri bulunamadı.")
    
    # Target 1 ve Target 2'yi belirle
    if targetNormalized > beforeNormalized:
        print("Değer yükselecek")
        target = [0, 1]
    else:
        print("Değer azalacak")
        target = [1, 0]
    
    csvData.append(inputData + target)

write_to_csv("coin.csv", csvData, len(csvData[0])-2, 2)
