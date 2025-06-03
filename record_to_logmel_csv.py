import sounddevice as sd
import librosa
import numpy as np
import pandas as pd
import os

# === Ayarlar ===
SAMPLING_RATE = 22050
DURATION = 1  # saniye
N_MELS = 40
CSV_FILE = "trainingDatas/voice_data.csv"

# 29 harf (Türk alfabesi için)
TURKISH_ALPHABET = [
    'a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü',
    'v', 'y', 'z',' '
]

def record_audio(duration=DURATION, samplerate=SAMPLING_RATE):
    print("🔴 Kayıt başladı...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Kayıt tamamlandı.")
    return audio.flatten()

def extract_logmel(audio, sr=SAMPLING_RATE, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)
    logmel = np.mean(logmel, axis=1)        # Zaman üzerinden ortalama
    logmel = (logmel - np.min(logmel)) / (np.max(logmel) - np.min(logmel))  # Normalize et
    logmel = np.round(logmel, 3)            # Yuvarla
    return logmel


def save_to_csv(logmel_vector, target_letter, csv_path=CSV_FILE):
    input_data = list(logmel_vector)
    target_data = [1 if letter == target_letter else 0 for letter in TURKISH_ALPHABET]
    row = input_data + target_data

    # Eğer dosya yoksa başlıkla birlikte yaz
    if not os.path.exists(csv_path):
        columns = [f"input{i+1}" for i in range(len(input_data))] + [f"target{i+1}" for i in range(len(target_data))]
        df = pd.DataFrame([row], columns=columns)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"💾 Veri kaydedildi: \"{target_letter}\"")


def main():
    repeatForEveryLetter=3
    repeat=0
    user_input = input("Bir harf yaz ve enter'a bas (çıkmak için q): ")
    while True:
        if repeat>=repeatForEveryLetter:
            user_input = input("Bir harf yaz ve enter'a bas (çıkmak için q): ")
            if len(user_input) != 1:
                print("⚠️ Lütfen sadece bir karakter gir.")
                continue


            if user_input not in TURKISH_ALPHABET:
                print("⚠️ Geçersiz harf. Lütfen Türk alfabesinden bir harf gir.")
                continue
            repeat = 0
        else:
            repeat = repeat + 1



        audio = record_audio()
        logmel_vector = extract_logmel(audio)
        save_to_csv(logmel_vector, user_input)

if __name__ == "__main__":
    main()
