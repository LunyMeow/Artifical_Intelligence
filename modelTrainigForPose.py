import cv2
import mediapipe as mp
import pandas as pd

# MediaPipe Pose modelini başlat
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Video kaynağını başlat (0 varsayılan kameradır)
cap = cv2.VideoCapture(0)

# Poz verilerini saklamak için liste
pose_data = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Görüntüyü RGB'ye çevir
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Poz tespiti başarılıysa
    if results.pose_landmarks:
        # Her bir landmark için x, y, z koordinatlarını al
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        pose_data.append(landmarks)

        # Pozu görselleştir
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Görüntüyü göster
    cv2.imshow('Pose Estimation', frame)

    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

# Poz verilerini DataFrame'e çevir
df = pd.DataFrame(pose_data)

# Giriş ve hedef verileri oluştur
# Her satırın hedefi, bir sonraki satırdır
inputs = df[:-1].reset_index(drop=True)
targets = df[1:].reset_index(drop=True)

# Sütun adlarını yeniden adlandır
inputs.columns = [f'input{i+1}' for i in range(inputs.shape[1])]
targets.columns = [f'target{i+1}' for i in range(targets.shape[1])]

# Giriş ve hedefleri birleştir
dataset = pd.concat([inputs, targets], axis=1)

# CSV dosyasına yaz
dataset.to_csv('pose_dataset.csv', index=False)
