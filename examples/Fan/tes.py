import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("v00_depth.mp4")

hist = np.zeros(256)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist += h.flatten()

cap.release()

plt.plot(hist)
plt.xlim([0,255])
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()