import cv2
import numpy as np
import pandas as pd
import os

def renk_filtrele(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def çizgi_eğimleri_bul(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eğimler = []
    for contour in contours:
        if len(contour) >= 2:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            slope = vy / vx
            eğimler.append(float(slope))
    return eğimler

def yönü_belirle(mask):
    # yeşil çizginin yönünü belirle
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 2:
        return "bilinmiyor"
    sorted_points = sorted(points, key=lambda p: p[0][1])  # y'ye göre sırala
    y1 = sorted_points[0][0][1]
    y2 = sorted_points[-1][0][1]
    return "yukarı" if y2 < y1 else "aşağı"

def görselleri_analiz_et(klasör):
    veriler = []
    for dosya in os.listdir(klasör):
        if dosya.endswith(".png") or dosya.endswith(".jpg"):
            yol = os.path.join(klasör, dosya)
            image = cv2.imread(yol)

            turuncu_mask = renk_filtrele(image, np.array([10, 100, 100]), np.array([25, 255, 255]))
            yeşil_mask = renk_filtrele(image, np.array([40, 50, 50]), np.array([80, 255, 255]))

            eğimler = çizgi_eğimleri_bul(turuncu_mask)
            hedef = yönü_belirle(yeşil_mask)

            giriş = {f"input{i+1}": e for i, e in enumerate(eğimler)}
            giriş["target1"] = 1 if hedef == "yukarı" else 0
            giriş["target2"] = 0 if hedef == "yukarı" else 1
            veriler.append(giriş)
    return pd.DataFrame(veriler)

# 🔁 Ana fonksiyonu çağır
csv_df = görselleri_analiz_et("Veriler")
csv_df.to_csv("grafik_formasyon_verisi.csv", index=False)
