import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from PIL import Image
import os

class DigitGenerator:
    def __init__(self):
        self.size = 8
        self.digits = {
            0: self.generate_zero,
            1: self.generate_one,
            2: self.generate_two,
            3: self.generate_three,
            4: self.generate_four,
            5: self.generate_five,
            6: self.generate_six,
            7: self.generate_seven,
            8: self.generate_eight,
            9: self.generate_nine
        }
        
    # Her rakam için özel çizim fonksiyonları
    def generate_zero(self):
        arr = np.zeros((self.size, self.size))
        for i in range(2, 6):
            arr[i][2] = 1
            arr[i][5] = 1
        for j in range(2, 6):
            arr[2][j] = 1
            arr[5][j] = 1
        return arr

    def generate_one(self):
        arr = np.zeros((self.size, self.size))
        for i in range(1, 7):
            arr[i][4] = 1
        arr[1][3] = 1
        arr[2][2] = 1
        return arr

    def generate_two(self):
        arr = np.zeros((self.size, self.size))
        arr[1][2:6] = 1
        arr[2][5] = 1
        arr[3][4] = 1
        arr[4][3] = 1
        arr[5][2] = 1
        arr[6][2:6] = 1
        return arr

    def generate_three(self):
        arr = np.zeros((self.size, self.size))
        arr[1][2:6] = 1
        arr[2][5] = 1
        arr[3][4] = 1
        arr[4][3:5] = 1
        arr[5][5] = 1
        arr[6][2:6] = 1
        return arr

    def generate_four(self):
        arr = np.zeros((self.size, self.size))
        for i in range(1, 5):
            arr[i][3] = 1
        arr[4][2:6] = 1
        for i in range(5, 7):
            arr[i][5] = 1
        return arr

    def generate_five(self):
        arr = np.zeros((self.size, self.size))
        arr[1][2:6] = 1
        arr[2][2] = 1
        arr[3][2:5] = 1
        arr[4][5] = 1
        arr[5][5] = 1
        arr[6][2:5] = 1
        return arr

    def generate_six(self):
        arr = np.zeros((self.size, self.size))
        arr[1][3:5] = 1
        arr[2][2] = 1
        arr[3][2] = 1
        arr[4][2:5] = 1
        arr[5][2] = 1
        arr[5][5] = 1
        arr[6][3:5] = 1
        return arr

    def generate_seven(self):
        arr = np.zeros((self.size, self.size))
        arr[1][2:6] = 1
        arr[2][5] = 1
        arr[3][4] = 1
        arr[4][3] = 1
        arr[5][3] = 1
        arr[6][3] = 1
        return arr

    def generate_eight(self):
        arr = np.zeros((self.size, self.size))
        arr[1][3:5] = 1
        arr[2][2] = 1
        arr[2][5] = 1
        arr[3][3:5] = 1
        arr[4][2] = 1
        arr[4][5] = 1
        arr[5][2] = 1
        arr[5][5] = 1
        arr[6][3:5] = 1
        return arr

    def generate_nine(self):
        arr = np.zeros((self.size, self.size))
        arr[1][3:5] = 1
        arr[2][2] = 1
        arr[2][5] = 1
        arr[3][2] = 1
        arr[3][5] = 1
        arr[4][3:6] = 1
        arr[5][5] = 1
        arr[6][3:5] = 1
        return arr

    def add_variations(self, digit_array):
        """Rakama çeşitlilik ve gürültü ekler"""
        for _ in range(3):
            i, j = random.randint(1,6), random.randint(1,6)
            digit_array[i][j] = 1 if random.random() > 0.5 else 0
        
        noise = np.random.rand(*digit_array.shape) * 0.3
        return np.clip(digit_array + noise, 0, 1)

    def generate_dataset(self, samples_per_digit=1):
        dataset = []
        for digit, generator in self.digits.items():
            for _ in range(samples_per_digit):
                clean_digit = generator()
                varied_digit = self.add_variations(clean_digit)
                
                # One-hot encoding (10 çıkış nöronu için)
                target = [0]*10
                target[digit] = 1
                
                dataset.append((varied_digit.flatten(), target))
        return dataset

    def save_to_csv(self, filename="digits_dataset.csv"):
        dataset = self.generate_dataset()
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Başlık satırı (input1...input64, target1...target10)
            header = [f"input{i+1}" for i in range(self.size*self.size)] + \
                     [f"target{i+1}" for i in range(10)]
            writer.writerow(header)
            
            for pixels, targets in dataset:
                writer.writerow(list(pixels) + list(targets))
        
        print(f"Toplam {len(dataset)} örnek '{filename}' dosyasına kaydedildi.")

    def visualize_samples(self, num_samples=5):
        """Rastgele örnekleri görselleştir"""
        plt.figure(figsize=(15, 3))
        dataset = self.generate_dataset()
        for i in range(num_samples):
            pixels, targets = random.choice(dataset)
            plt.subplot(1, num_samples, i+1)
            plt.imshow(pixels.reshape(self.size, self.size), cmap='gray')
            plt.title(f"Target: {np.argmax(targets)}")
            plt.axis('off')
        plt.show()

# Programı çalıştır
if __name__ == "__main__":
    generator = DigitGenerator()
    
    print("Rakam örnekleri görselleştiriliyor...")
    generator.visualize_samples()
    
    print("Veri seti oluşturuluyor...")
    generator.save_to_csv()
    
    print("\nCSV dosyası başarıyla oluşturuldu. Aşağıdaki komutla eğitime başlayabilirsiniz:")
    print("train_custom('digits_dataset.csv')")