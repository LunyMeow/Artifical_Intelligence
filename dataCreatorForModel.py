import random
import pygame
import numpy as np
import csv
import sys

# Ayarlar
GRID_SIZE = 16
PIXEL_SIZE = 32  # Ekranda görünen boyut
WIDTH, HEIGHT = GRID_SIZE * PIXEL_SIZE, GRID_SIZE * PIXEL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BG_COLOR = WHITE
DRAW_COLOR = BLACK

# Pygame başlat
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rakam Çizici - 16x16")
clock = pygame.time.Clock()

# Boş grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# CSV dosyası
csv_filename = "digit_data.csv"

# Başlık varsa yazma, yoksa yaz
import os
if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"input{i+1}" for i in range(256)] + [f"target{i+1}" for i in range(10)]
        writer.writerow(header)


def save_to_csv(label):
    flat_pixels = grid.flatten().tolist()
    one_hot_target = [1 if i == label else 0 for i in range(10)]
    row = flat_pixels + one_hot_target
    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"Kaydedildi: {label}")

def draw_grid():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color = DRAW_COLOR if grid[y][x] == 1 else BG_COLOR
            pygame.draw.rect(screen, color, (x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
            pygame.draw.rect(screen, (200, 200, 200), (x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE), 1)

def reset_grid():
    global grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

def get_mouse_pos():
    x, y = pygame.mouse.get_pos()
    return x // PIXEL_SIZE, y // PIXEL_SIZE



def randomOne():
    global grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    offset_x = random.randint(5, 9)  # Dikey çizgi pozisyonu
    slant = random.choice([-1, 0, 1])  # Eğiklik için başlangıç çizgisi yönü

    # Başlangıç eğik çizgisi (şapka gibi)
    grid[2][offset_x + slant] = 1
    grid[3][offset_x] = 1
    grid[4][offset_x] = 1

    # Dikey gövde (gövde boyunca küçük varyasyonlar)
    for y in range(5, 13):
        dx = random.choice([-1, 0, 0, 0, 1])  # Ağırlıklı olarak düz ama az varyasyonlu
        x_pos = offset_x + dx
        if 0 <= x_pos < GRID_SIZE:
            grid[y][x_pos] = 1

    # Bitiş çizgisi (küçük ayak gibi)
    grid[13][offset_x - 1] = 1
    grid[13][offset_x] = 1
    grid[13][offset_x + 1] = 1

    save_to_csv(1)
    print("Rastgele ve gerçekçi 1 çizildi ve kaydedildi.")






def main():
    current_label = None
    drawing = False

    while True:
        screen.fill(BG_COLOR)
        draw_grid()
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Tuşla rakam seçimi
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                                 pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                    current_label = int(event.unicode)
                    print(f"Seçilen rakam: {current_label}")

                # Kaydetme tuşu (s)
                elif event.key == pygame.K_s and current_label is not None:
                    save_to_csv(current_label)

                # Temizleme tuşu (c)
                elif event.key == pygame.K_c:
                    reset_grid()
                    print("Palet temizlendi.")
                elif event.key == pygame.K_r:
                    randomOne()

            # Çizim başla
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

        if drawing:
            mx, my = get_mouse_pos()
            if 0 <= mx < GRID_SIZE and 0 <= my < GRID_SIZE:
                grid[my][mx] = 1

        clock.tick(60)

if __name__ == "__main__":
    main()
