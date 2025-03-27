import cv2
import numpy as np
import os

data_dir = "training_data"
os.makedirs(data_dir, exist_ok=True)

def save_drawing(label):
    img = np.ones((200, 200), dtype=np.uint8) * 255
    drawing = False

    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(img, (x, y), 8, (0,), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw a number")
    cv2.setMouseCallback("Draw a number", draw)
    
    while True:
        cv2.imshow("Draw a number", img)
        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            break
    
    cv2.destroyAllWindows()
    filename = os.path.join(data_dir, f"{label}_{len(os.listdir(data_dir))}.png")
    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    while True:
        label = input("Enter the number you drew (0-9) or 'q' to quit: ")
        if label.lower() == 'q':
            break
        if label.isdigit() and 0 <= int(label) <= 9:
            save_drawing(label)
        else:
            print("Invalid input. Please enter a number between 0 and 9.")