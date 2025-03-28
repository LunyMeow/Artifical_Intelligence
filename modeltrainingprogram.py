import csv
import random
from typing import List, Tuple

def read_csv_file(file_path: str) -> Tuple[List[List[float]], List[List[int]]]:
    """
    CSV dosyasından verileri dinamik olarak okur
    Args:
        file_path: CSV dosya yolu
    Returns:
        X_train: Giriş verileri listesi
        y_train: Hedef verileri listesi
    """
    X_train = []
    y_train = []
    
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)  # Başlık satırını oku
        
        # Giriş ve hedef sütunlarını otomatik ayır
        input_cols = [col for col in headers if col.startswith('input')]
        target_cols = [col for col in headers if col.startswith('target')]
        
        input_indices = [headers.index(col) for col in input_cols]
        target_indices = [headers.index(col) for col in target_cols]
        
        for row in csv_reader:
            # Giriş verilerini float olarak oku
            inputs = [float(row[i]) for i in input_indices]
            # Hedef verilerini int olarak oku
            targets = [int(row[i]) for i in target_indices]
            
            X_train.append(inputs)
            y_train.append(targets)
    
    return X_train, y_train

def write_csv_file(file_path: str, X_data: List[List[float]], y_data: List[List[int]]):
    """
    CSV dosyasına verileri dinamik olarak yazar
    Args:
        file_path: Kaydedilecek CSV dosya yolu
        X_data: Giriş verileri
        y_data: Hedef verileri
    """
    # Sütun başlıklarını otomatik oluştur
    num_inputs = len(X_data[0]) if X_data else 0
    num_targets = len(y_data[0]) if y_data else 0
    
    input_headers = [f'input{i+1}' for i in range(num_inputs)]
    target_headers = [f'target{i+1}' for i in range(num_targets)]
    headers = input_headers + target_headers
    
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Başlıkları yaz
        writer.writerow(headers)
        
        # Verileri yaz
        for inputs, targets in zip(X_data, y_data):
            row = inputs + targets
            writer.writerow(row)


def generate_random_data(num_samples: int) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Rastgele veri üretir
    Args:
        num_samples: Üretilecek örnek sayısı
    Returns:
        X_data: Giriş verileri (List[List[float]])
        y_data: Hedef verileri (List[List[int]]) (one-hot encoded)
    """
    X_data = []
    y_data = []
    
    for _ in range(num_samples):
        # Rastgele 5 input değeri (0-1 arası, 2 ondalık basamaklı)
        inputs = [round(random.uniform(0, 1), 2) for _ in range(5)]
        
        # Rastgele bir target seç (one-hot encoding)
        target_class = random.randint(0, 2)
        targets = [1 if i == target_class else 0 for i in range(3)]
        
        X_data.append(inputs)
        y_data.append(targets)
    
    return X_data, y_data


# Kullanım örneği
if __name__ == "__main__":

    # Veri üret
    X, y = generate_random_data(100)

# CSV'ye yaz
    write_csv_file('random_data.csv', X, y)
    
    # CSV'den okuma
    X_loaded, y_loaded = read_csv_file('ornek_veri.csv')
    print("\nOkunan veriler:")
    print("Girişler:", X_loaded)
    print("Hedefler:", y_loaded)