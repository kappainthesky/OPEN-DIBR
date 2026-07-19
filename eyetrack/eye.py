# Implements the 4-stage pipeline:
#   COLOR FRAME  -> IRIS LANDMARKS -> ALIGNED DEPTH -> 3D PROJECTION

import sys
import subprocess
import os

# Check if we have the correct Python version and dependencies
try:
    import cv2 as cv
    import numpy as np
    import mediapipe as mp
    import pyrealsense2 as rs
    _ = mp.solutions.face_mesh
except (ImportError, AttributeError):
    # Try to find a compatible Python interpreter on the system (e.g. Python 3.11)
    candidates = [
        r"C:\Users\Syncard\AppData\Local\Programs\Python\Python311\python.exe",
        r"C:\Users\Syncard\AppData\Local\Programs\Python\Python313\python.exe",
        r"C:\Python314\python.exe",
    ]
    try:
        output = subprocess.check_output(["where", "python"], text=True)
        for line in output.strip().splitlines():
            line = line.strip()
            if line and line not in candidates:
                candidates.append(line)
    except Exception:
        pass

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        if os.path.abspath(candidate) == os.path.abspath(sys.executable):
            continue
        try:
            test_cmd = [candidate, "-c", "import cv2; import numpy; import mediapipe as mp; _ = mp.solutions.face_mesh; import pyrealsense2"]
            subprocess.check_call(test_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Launcher] Current Python ({sys.version.split()[0]}) is missing dependencies or incompatible.")
            print(f"[Launcher] Switching to compatible Python: {candidate}")
            result = subprocess.run([candidate] + sys.argv)
            sys.exit(result.returncode)
        except Exception:
            continue
            
    print("\n[Error] Could not find a Python environment with all required dependencies:")
    print("  - opencv-python")
    print("  - numpy")
    print("  - mediapipe")
    print("  - pyrealsense2")
    print("\nPlease run: pip install opencv-python numpy mediapipe pyrealsense2\n")
    sys.exit(1)

import time
import json
from datetime import datetime
import socket

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# D455 rated range — depth is unreliable closer than this
MIN_VALID_DEPTH_MM = 300
MAX_VALID_DEPTH_MM = 6000


# RealSense setup
USE_WEBCAM = False
pipeline = None
align = None
cap = None

try:
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile  = pipeline.start(config)

    # depth to color alignment so depth[u,v] lines up with the color pixel
    align = rs.align(rs.stream.color)

    # Use the color stream's OWN factory-calibrated intrinsics, since iris
    # landmarks are detected on the color frame.
    color_intr = profile.get_stream(rs.stream.color) \
                         .as_video_stream_profile().get_intrinsics()
    fx, fy = color_intr.fx, color_intr.fy
    cx, cy = color_intr.ppx, color_intr.ppy
except RuntimeError as e:
    print(f"\n[RealSense Setup Error]: {e}")
    print("Falling back to standard Webcam (cv2.VideoCapture)...")
    USE_WEBCAM = True
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
         print("Error: Could not open standard Webcam!")
    # Standard webcam approximation for intrinsics
    fx, fy = 600.0, 600.0
    cx, cy = 320.0, 240.0


CAPTURE_WIDTH = 640
cx = (CAPTURE_WIDTH - 1) - cx

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_calib.json")

def load_screen_calibration(path):
    try:
        with open(path, 'r') as f:
            calib = json.load(f)
        R = np.array(calib['rotation_matrix'], dtype=np.float32)
        t_dict = calib['translation_mm']
        t = np.array([t_dict['x'], t_dict['y'], t_dict['z']], dtype=np.float32)
        if R.shape != (3, 3):
            raise ValueError(f"rotation_matrix must be 3x3, got {R.shape}")
        return R, t, True
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"\n[screen calibration] '{path}' not found or invalid ({e}); "
              f"using identity transform — midpoint_screen == midpoint\n")
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), False

SCREEN_R, SCREEN_T, SCREEN_CALIBRATED = load_screen_calibration(CALIB_FILE)

# UDP Socket setup for sending eye coordinates to OpenDIBR C++ application
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def camera_to_screen(p_cam):
    """Apply the camera -> screen rigid transform: R @ p + t."""
    return SCREEN_R @ p_cam + SCREEN_T


print(f"\nResolution : 640 x 480")
print(f"\nfx = {fx:.2f}   fy = {fy:.2f}")
print(f"\ncx (mirrored) = {cx:.1f}   cy = {cy:.1f}   "
      f"(frame visual center would be 320.0, 240.0)")
print(f"\nScreen calibration: "
      f"{'loaded from ' + CALIB_FILE if SCREEN_CALIBRATED else 'NOT applied (identity)'}")
print(f"\nQ = quit\n")

# session recording for JSON export
session_start_unix = time.time()
frame_log = []
frame_idx = 0


# Stage 3: aligned depth

def get_aligned_depth(depth_img, u, v, patch=3):
    """
    Median depth in a small SYMMETRIC patch around (u, v) in the
    color-aligned depth frame (patch=3 -> 7x7 window, centered exactly
    on the pixel). Using a patch instead of a single pixel avoids
    holes/noise in the raw depth map.
    Returns 0.0 if no valid depth is found (e.g. IR-absorbing surface,
    out of range, or a depth hole).
    """
    u, v = int(u), int(v)
    h, w = depth_img.shape
    u = max(patch, min(w - patch - 1, u))
    v = max(patch, min(h - patch - 1, v))
    region = depth_img[v - patch:v + patch + 1, u - patch:u + patch + 1]
    valid = region[region > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


# stage 4: 3D projection

def pixel_to_3d(u, v, depth_mm):
    x_mm = (u - cx) / fx * depth_mm
    y_mm = (v - cy) / fy * depth_mm
    z_mm = depth_mm
    return np.array([x_mm, y_mm, z_mm], dtype=np.float32)


# Smoothing

class Smoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.val = None

    def update(self, new_val):
        if self.val is None:
            self.val = new_val.copy()
        else:
            self.val = self.alpha * new_val + (1 - self.alpha) * self.val
        return self.val.copy()


mid_smoother = Smoother(alpha=0.35)


# Pipeline output hook

def emit_eye_position(eye_mid_mm, timestamp):
    """
    Single hook point where the eye position gets handed off to the
    rest of the windowed-6DoF pipeline (OpenDIBR / 2DGS renderer).
    Sends the x, y, z coordinates in mm to OpenDIBR over UDP.
    """
    x, y, z = eye_mid_mm
    print(
        f"[{timestamp:.3f}] EYE x={x:+7.1f} y={y:+7.1f} z={z:7.1f} mm",
        end='\r'
    )
    try:
        msg = f"{x} {y} {z}"
        sock.sendto(msg.encode('utf-8'), (UDP_IP, UDP_PORT))
    except Exception as e:
        pass


# Main loop

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Running — press Q to quit")

    while True:
        if USE_WEBCAM:
            if cap is None or not cap.isOpened():
                print("\n[Error] Webcam is not opened. Cannot capture frames. Exiting.")
                break
            ret, color_img = cap.read()
            if not ret:
                print("Warning: Failed to capture image from webcam. Retrying in 1 second...", end="\r")
                time.sleep(1.0)
                continue
            # Since we don't have a depth sensor, create a mock depth image (filled with 600 mm)
            depth_img = np.full((480, 640), 600, dtype=np.uint16)
        else:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

        # flip both consistently so pixel coords line up after the flip
        frame     = cv.flip(color_img, 1)
        depth_flp = np.fliplr(depth_img)

        img_h, img_w = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results   = face_mesh.process(rgb_frame)
        display   = frame.copy()

        if results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])

            # Stage 2: iris landmarks
            # (l_r / r_r are enclosing-circle radii, used only for the
            # on-screen overlay below — not logged to the session JSON)
            (l_u, l_v), l_r = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_u, r_v), r_r = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left  = np.array([l_u, l_v], dtype=np.int32)
            center_right = np.array([r_u, r_v], dtype=np.int32)

            cv.circle(display, center_left,  int(l_r), (255, 0, 0), 2, cv.LINE_AA)
            cv.circle(display, center_right, int(r_r), (255, 0, 0), 2, cv.LINE_AA)
            cv.circle(display, center_left,  1, (0, 0, 255), -1, cv.LINE_AA)
            cv.circle(display, center_right, 1, (0, 0, 255), -1, cv.LINE_AA)

            # Stage 3: aligned depth (from D455 sensor, not estimated)
            l_depth_raw = get_aligned_depth(depth_flp, l_u, l_v)
            r_depth_raw = get_aligned_depth(depth_flp, r_u, r_v)

            valid_depths = [d for d in (l_depth_raw, r_depth_raw)
                             if MIN_VALID_DEPTH_MM <= d <= MAX_VALID_DEPTH_MM]

            if len(valid_depths) == 0:
                cv.putText(display, "No valid depth (too close / out of range)",
                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                l_depth = l_depth_raw if l_depth_raw > 0 else np.median(valid_depths)
                r_depth = r_depth_raw if r_depth_raw > 0 else np.median(valid_depths)

                # Stage 4: 3D projection
                l_3d = pixel_to_3d(l_u, l_v, l_depth)
                r_3d = pixel_to_3d(r_u, r_v, r_depth)
                eye_mid_raw = (l_3d + r_3d) / 2.0
                eye_mid = mid_smoother.update(eye_mid_raw)

                cv.putText(display,
                    f"L: x={l_3d[0]:+.0f} y={l_3d[1]:+.0f} z={l_3d[2]:.0f} mm",
                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                cv.putText(display,
                    f"R: x={r_3d[0]:+.0f} y={r_3d[1]:+.0f} z={r_3d[2]:.0f} mm",
                    (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                cv.putText(display,
                    f"MID: x={eye_mid[0]:+.0f} y={eye_mid[1]:+.0f} z={eye_mid[2]:.0f} mm",
                    (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(display,
                    f"Frames logged: {frame_idx}",
                    (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                emit_eye_position(eye_mid, time.time())

                # Stage 5: camera -> screen transform (identity until calibrated)
                eye_mid_screen = camera_to_screen(eye_mid)

                # log this frame for JSON export
                frame_log.append({
                    'frame': frame_idx,
                    't_sec': round(time.time() - session_start_unix, 3),
                    'left_eye': {
                        'x': float(l_3d[0]), 'y': float(l_3d[1]), 'z': float(l_3d[2])
                    },
                    'right_eye': {
                        'x': float(r_3d[0]), 'y': float(r_3d[1]), 'z': float(r_3d[2])
                    },
                    'midpoint': {
                        'x': float(eye_mid[0]), 'y': float(eye_mid[1]), 'z': float(eye_mid[2])
                    },
                    'midpoint_screen': {
                        'x': float(eye_mid_screen[0]), 'y': float(eye_mid_screen[1]), 'z': float(eye_mid_screen[2])
                    }
                })
                frame_idx += 1
        else:
            cv.putText(display, "No face detected",
                (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.imshow('Eye 3D Position (RGB-D / D455)', display)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

if USE_WEBCAM:
    if cap is not None:
        cap.release()
else:
    if pipeline is not None:
        pipeline.stop()
cv.destroyAllWindows()

# save session JSON
session_duration_s = round(time.time() - session_start_unix, 2)
timestamp_str = datetime.fromtimestamp(session_start_unix).strftime('%Y%m%d_%H%M%S')
output_filename = f'eye_tracking_3d_rgbd_{timestamp_str}.json'

session_data = {
    'session_start_unix': session_start_unix,
    'session_duration_s': session_duration_s,
    'depth_source': 'RealSense D455 aligned depth sensor',
    'camera_intrinsics': {
        'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy)
    },
    'screen_calibration': {
        'calibrated': SCREEN_CALIBRATED,
        'rotation_matrix': SCREEN_R.tolist(),
        'translation_mm': {
            'x': float(SCREEN_T[0]), 'y': float(SCREEN_T[1]), 'z': float(SCREEN_T[2])
        },
        'note': ('loaded from ' + CALIB_FILE) if SCREEN_CALIBRATED
                 else 'no screen calibration applied — identity transform, camera-space coordinates'
    },
    'resolution': {'width': CAPTURE_WIDTH, 'height': 480},
    'num_frames': len(frame_log),
    'frames': frame_log
}

with open(output_filename, 'w') as f:
    json.dump(session_data, f, indent=2)

print(f"\nSaved {len(frame_log)} frames to {output_filename}")
print("Done.")