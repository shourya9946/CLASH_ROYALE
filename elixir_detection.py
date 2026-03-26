import cv2
import numpy as np

img = cv2.imread("clash.jpeg")
if img is None:
    print("Image not found")
    exit()

h, w, _ = img.shape
roi = img[int(h*0.8):h, 0:w]

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

lower_purple = np.array([130, 80, 80])
upper_purple = np.array([170, 255, 255])

mask = cv2.inRange(hsv, lower_purple, upper_purple)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

elixir = 0

if contours:
    largest = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest)

    cv2.rectangle(img, (x, y + int(h*0.8)), (x+w_box, y+h_box + int(h*0.8)), (0,255,0), 2)

    elixir = int((w_box / (w * 0.7)) * 10)
    elixir = max(0, min(10, elixir))

print("Elixir:", elixir)

cv2.putText(img, f"Elixir: {elixir}", (50, h-50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

cv2.imshow("Detection", img)
cv2.imshow("Mask", mask)

while True:
    key = cv2.waitKey(10) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()