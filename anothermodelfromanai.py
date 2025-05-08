import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback

# Veri setini yükle
data = [
    [0,0,0,0,1,0],
    [0,0,0,1,0,1],
    [0,0,1,0,0,1],
    [0,0,1,1,1,0],
    [0,1,0,0,1,0],
    [0,1,0,1,1,0],
    [0,1,1,0,1,0],
    [0,1,1,1,0,1],
    [1,0,0,0,0,1],
    [1,0,0,1,1,0],
    [1,0,1,0,1,0],
    [1,0,1,1,0,1],
    [1,1,0,0,1,0],
    [1,1,0,1,0,1],
    [1,1,1,0,0,1],
    [1,1,1,1,1,0]
]
print("Debug1")
df = pd.DataFrame(data, columns=['input1','input2','input3','input4','target1','target2'])
X = df[['input1','input2','input3','input4']].values
y = df[['target1','target2']].values

# Özel callback sınıfı
class TimeTrackingCallback(Callback):
    def __init__(self, target_loss):
        super().__init__()
        self.target_loss = target_loss
        self.start_time = None
        self.reached_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.target_loss and self.reached_time is None:
            self.reached_time = time.time() - self.start_time
            self.model.stop_training = True

# Modeli oluştur
print("Debug2")

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Eğitim parametreleri
target_loss = 0.0009
callback = TimeTrackingCallback(target_loss)

print("Debug3")
# Eğitim süresini ölç
start_time = time.time()
history = model.fit(X, y, 
                    epochs=1000, 
                    batch_size=2, 
                    verbose=0, 
                    callbacks=[callback])

# Sonuçları göster
print("Debug4")
training_time = time.time() - start_time
final_loss = history.history['loss'][-1]
reached_time = callback.reached_time if callback.reached_time else training_time

print(f"Son hata değeri: {final_loss:.6f}")
print(f"%99 doğruluğa ulaşma süresi: {reached_time:.4f} saniye")
print(f"Toplam eğitim süresi: {training_time:.4f} saniye")