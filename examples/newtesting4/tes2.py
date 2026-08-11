import cv2
import numpy as np

input_video = "v00_depth(backup).mp4"
output_video = "v00_depth_grayscale_filtered.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise RuntimeError("Cannot open input video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height),
    True
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only valid when the source already represents grayscale/inverse depth.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove isolated noise while preserving most depth boundaries.
    filtered = cv2.medianBlur(gray, 3)

    # Keep the full 8-bit range.
    output = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    writer.write(output)

cap.release()
writer.release()

print("Finished!")