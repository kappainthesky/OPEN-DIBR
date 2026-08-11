import cv2
import numpy as np

# Input dan output video
input_video = "v00_depth.mp4"
output_video = "v00_depth_reduced_white.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise RuntimeError("Cannot open input video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height), True)

# Target maksimum intensitas
TARGET_MAX = 55

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Median filter untuk mengurangi noise
    gray = cv2.medianBlur(gray, 3)

    # Compress intensity range
    corrected = np.clip(
        gray.astype(np.float32) * (TARGET_MAX / 255.0),
        0,
        TARGET_MAX
    ).astype(np.uint8)

    corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)

    writer.write(corrected_bgr)

cap.release()
writer.release()

print("Finished!")