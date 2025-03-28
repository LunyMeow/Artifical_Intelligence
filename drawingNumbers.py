import numpy as np
import csv
import os
from PIL import Image, ImageDraw
import random

def generate_digit_image(digit, size=(28, 28), noise_level=0.1):
    """Rastgele bir rakam çizimi oluştur"""
    img = Image.new('L', size, color=0)  # Siyah arka plan
    draw = ImageDraw.Draw(img)
    
    # Rakamın başlangıç pozisyonu ve boyutu
    width, height = size
    digit_size = random.randint(10, min(20, width-10, height-10))  # Görüntüyü aşmayacak şekilde
    
    # Pozisyonu güvenli aralıkta hesapla
    max_x = max(5, width - digit_size - 5)
    max_y = max(5, height - digit_size - 5)
    
    x = random.randint(5, max_x)
    y = random.randint(5, max_y)
    
    # Rakam çiz (basit şekiller)
    if digit == 0:
        draw.ellipse([x, y, x+digit_size, y+digit_size], outline=255, width=2)
    elif digit == 1:
        draw.line([x+digit_size//2, y, x+digit_size//2, y+digit_size], fill=255, width=2)
    elif digit == 2:
        draw.arc([x, y, x+digit_size, y+digit_size], 30, 270, fill=255, width=2)
    elif digit == 3:
        draw.arc([x, y, x+digit_size, y+digit_size//2], 200, 340, fill=255, width=2)
        draw.arc([x, y+digit_size//2, x+digit_size, y+digit_size], 200, 340, fill=255, width=2)
    elif digit == 4:
        draw.line([x, y+digit_size//2, x+digit_size, y+digit_size//2], fill=255, width=2)
        draw.line([x+digit_size//2, y, x+digit_size//2, y+digit_size], fill=255, width=2)
    elif digit == 5:
        draw.line([x, y, x+digit_size, y], fill=255, width=2)
        draw.line([x, y, x, y+digit_size//2], fill=255, width=2)
        draw.line([x, y+digit_size//2, x+digit_size, y+digit_size//2], fill=255, width=2)
        draw.line([x+digit_size, y+digit_size//2, x+digit_size, y+digit_size], fill=255, width=2)
        draw.line([x, y+digit_size, x+digit_size, y+digit_size], fill=255, width=2)
    elif digit == 6:
        draw.ellipse([x, y, x+digit_size, y+digit_size], outline=255, width=2)
        draw.line([x+digit_size//2, y+digit_size//2, x+digit_size//2, y+digit_size-5], fill=255, width=2)
    elif digit == 7:
        draw.line([x, y, x+digit_size, y], fill=255, width=2)
        draw.line([x+digit_size, y, x, y+digit_size], fill=255, width=2)
    elif digit == 8:
        draw.ellipse([x, y, x+digit_size, y+digit_size//2], outline=255, width=2)
        draw.ellipse([x, y+digit_size//2, x+digit_size, y+digit_size], outline=255, width=2)
    elif digit == 9:
        draw.ellipse([x, y, x+digit_size, y+digit_size], outline=255, width=2)
        draw.line([x+digit_size//2, y, x+digit_size//2, y+digit_size//2], fill=255, width=2)
    
    # Gürültü ekle (gerçekçilik için)
    img_array = np.array(img)
    noise = np.random.randint(0, 50, size=img_array.shape) * (np.random.random(size=img_array.shape)) < noise_level
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return img_array

def create_dataset(num_samples_per_digit=200, output_file='minst.csv'):
    """MNIST benzeri bir dataset oluştur ve CSV'ye kaydet"""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Başlık satırı (input1...input784, target0...target9)
        headers = [f'input{i}' for i in range(784)] + [f'target{i}' for i in range(10)]
        writer.writerow(headers)
        
        for digit in range(10):
            print(f"Generating {num_samples_per_digit} samples for digit {digit}...")
            for _ in range(num_samples_per_digit):
                # Rakam görüntüsü oluştur
                img_array = generate_digit_image(digit)
                
                # Normalize et (0-1 arası)
                img_data = img_array.flatten() / 255.0
                
                # One-hot encoding için hedef
                target = [0] * 10
                target[digit] = 1
                
                # Satırı yaz
                writer.writerow(list(img_data) + target)
    
    print(f"Dataset created successfully: {output_file}")

if __name__ == "__main__":
    # 5000 örnek oluştur (her rakam için 500)
    create_dataset(num_samples_per_digit=50)