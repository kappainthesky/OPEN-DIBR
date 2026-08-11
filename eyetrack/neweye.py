# Implements the 4-stage pipeline:
#   COLOR FRAME  -> IRIS LANDMARKS -> ALIGNED DEPTH -> 3D PROJECTION

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import time
import json
from datetime import datetime
import socket
import os

# UDP Socket setup for sending eye coordinates to OpenDIBR C++ application
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Ring landmarks (4 pts each), used only for the on-screen radius overlay.
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# MediaPipe's own dedicated iris-center landmarks (single point each,
# not derived from the ring). Index pairing follows the ring convention
# already used above: 473 is the center of the 474-477 ring, 468 is the
# center of the 469-472 ring.
LEFT_IRIS_CENTER  = 473
RIGHT_IRIS_CENTER = 468

# D455 rated range — depth is unreliable closer than this
MIN_VALID_DEPTH_MM = 300
MAX_VALID_DEPTH_MM = 6000


# RealSense setup

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
profile  = pipeline.start(config)

# depth to color alignment so depth[u,v] lines up with the color pixel
align = rs.align(rs.stream.color)

# Use the color stream's OWN factory-calibrated intrinsics, since iris
# landmarks are detected on the color frame. color_intr carries the
# distortion coefficients too (coeffs/model), which rs2_deproject_pixel_to_point
# uses — the manual pinhole formula used previously did not, and was
# found (via test_x_accuracy.py) to introduce growing error off-axis.
color_intr = profile.get_stream(rs.stream.color) \
                     .as_video_stream_profile().get_intrinsics()
fx, fy = color_intr.fx, color_intr.fy
cx, cy = color_intr.ppx, color_intr.ppy

CAPTURE_WIDTH = 640

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


def camera_to_screen(p_cam):
    """Apply the camera -> screen rigid transform: R @ p + t."""
    return SCREEN_R @ p_cam + SCREEN_T


print(f"\nResolution : 640 x 480")
print(f"\nfx = {fx:.2f}   fy = {fy:.2f}   cx = {cx:.1f}   cy = {cy:.1f}   "
      f"(frame visual center would be 320.0, 240.0)")
print(f"\nDistortion : model={color_intr.model} coeffs={color_intr.coeffs}")
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
    """
    Deproject a pixel in the FLIPPED (mirror/display) frame to a 3D
    point in mirror-view coordinates: +x = user's right, +y = down,
    +z = depth away from the camera.

    Uses rs.rs2_deproject_pixel_to_point with the color stream's
    factory intrinsics (including distortion coefficients), rather
    than a manual pinhole formula, since distortion was confirmed to
    be non-negligible off-axis.
    """
    u = CAPTURE_WIDTH - 1 - u      # undo horizontal flip -> native camera-frame pixel coords

    point = rs.rs2_deproject_pixel_to_point(
        color_intr,
        [float(u), float(v)],
        float(depth_mm)
    )
    point = np.array(point, dtype=np.float32)

    point[0] = -point[0]

    return point


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


class LatencyPredictor:
    def __init__(self, latency_sec=0.045):
        self.latency = latency_sec
        self.pos = None
        self.last_smoothed_pos = None
        self.vel = np.zeros(3, dtype=np.float32)
        self.last_t = None

    def update(self, new_pos, t):
        if self.pos is None:
            self.pos = new_pos.copy()
            self.last_smoothed_pos = new_pos.copy()
            self.last_t = t
            return self.pos

        dt = t - self.last_t
        if dt <= 0:
            return self.pos

        # 1. Faster low-pass filter update for lower latency
        self.pos = 0.25 * self.pos + 0.75 * new_pos

        # 2. Estimate velocity from current and previous smoothed positions
        instant_vel = (self.pos - self.last_smoothed_pos) / dt
        self.last_smoothed_pos = self.pos.copy()

        # Jitter Gate: Lower threshold to prevent lag during slow head movements
        vel_mag = np.linalg.norm(instant_vel)
        if vel_mag < 3.0:
            instant_vel = np.zeros(3, dtype=np.float32)

        # 3. Faster velocity vector smoothing response
        self.vel = 0.55 * self.vel + 0.45 * instant_vel
        self.last_t = t

        # 4. Extrapolate position to compensate for pipeline latency
        predicted_pos = self.pos + self.vel * self.latency
        return predicted_pos


mid_smoother = LatencyPredictor(latency_sec=0.045)


# Pipeline output hook

def emit_eye_position(eye_mid_mm, state, timestamp):
    """
    Single hook point where the eye position gets handed off to the
    rest of the windowed-6DoF pipeline (OpenDIBR / 2DGS renderer).
    Sends the x, y, z coordinates in mm and the tracking state (0-3) to OpenDIBR over UDP.
    """
    x, y, z = eye_mid_mm
    print(
        f"[{timestamp:.3f}] EYE x={x:+7.1f} y={y:+7.1f} z={z:7.1f} mm | State={state}",
        end='\r'
    )
    try:
        msg = f"{x} {y} {z} {state}"
        sock.sendto(msg.encode('utf-8'), (UDP_IP, UDP_PORT))
    except Exception as e:
        pass


# State tracking globals
last_stable_pos = np.zeros(3, dtype=np.float32)
startup_frames = 0

# Main loop

with vision.FaceLandmarker.create_from_options(options) as landmarker:

    print("Running — press Q to quit")

    while True:
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results   = landmarker.detect(mp_image)
        display   = frame.copy()

        # Evaluate tracking state
        state = 0 # default TRACKING

        if not results.face_landmarks:
            state = 2 # LOST
            emit_eye_position(last_stable_pos, 2, time.time())
            cv.putText(display, "STATE: LOST (No face detected)",
                (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv.LINE_AA)
        else:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.face_landmarks[0]
            ])

            l_u, l_v = mesh_points[LEFT_IRIS_CENTER]
            r_u, r_v = mesh_points[RIGHT_IRIS_CENTER]

            _, l_r = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            _, r_r = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            # 1. Iris detection check (confidence metric)
            if l_r < 1.5 or r_r < 1.5 or l_r > 25.0 or r_r > 25.0:
                state = 1 # LOW_CONFIDENCE
                cv.putText(display, "STATE: LOW_CONFIDENCE (Iris anomaly)",
                    (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv.LINE_AA)
                emit_eye_position(last_stable_pos, 1, time.time())
            else:
                center_left  = np.array([l_u, l_v], dtype=np.int32)
                center_right = np.array([r_u, r_v], dtype=np.int32)

                cv.circle(display, center_left,  int(l_r), (255, 0, 0), 2, cv.LINE_AA)
                cv.circle(display, center_right, int(r_r), (255, 0, 0), 2, cv.LINE_AA)
                cv.circle(display, center_left,  1, (0, 0, 255), -1, cv.LINE_AA)
                cv.circle(display, center_right, 1, (0, 0, 255), -1, cv.LINE_AA)

                l_depth_raw = get_aligned_depth(depth_flp, l_u, l_v)
                r_depth_raw = get_aligned_depth(depth_flp, r_u, r_v)

                valid_depths = [d for d in (l_depth_raw, r_depth_raw)
                                 if MIN_VALID_DEPTH_MM <= d <= MAX_VALID_DEPTH_MM]

                # 2. Depth validity check
                if len(valid_depths) == 0:
                    state = 1 # LOW_CONFIDENCE
                    cv.putText(display, "STATE: LOW_CONFIDENCE (Invalid depth)",
                        (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv.LINE_AA)
                    emit_eye_position(last_stable_pos, 1, time.time())
                else:
                    l_depth = l_depth_raw if l_depth_raw > 0 else np.median(valid_depths)
                    r_depth = r_depth_raw if r_depth_raw > 0 else np.median(valid_depths)

                    l_3d = pixel_to_3d(l_u, l_v, l_depth)
                    r_3d = pixel_to_3d(r_u, r_v, r_depth)
                    eye_mid_raw = (l_3d + r_3d) / 2.0

                    # 3. Coordinate Jump check
                    jump_dist = np.linalg.norm(eye_mid_raw - last_stable_pos)
                    if startup_frames > 20 and jump_dist > 120.0:
                        state = 1 # LOW_CONFIDENCE
                        cv.putText(display, "STATE: LOW_CONFIDENCE (Coordinate jump)",
                            (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv.LINE_AA)
                        emit_eye_position(last_stable_pos, 1, time.time())
                    else:
                        # Smooth coordinate tracking
                        eye_mid = mid_smoother.update(eye_mid_raw, time.time())
                        last_stable_pos = eye_mid.copy()

                        # 4. Startup recalibration check
                        if startup_frames < 20:
                            state = 3 # RECALIBRATING
                            startup_frames += 1
                            cv.putText(display, "STATE: RECALIBRATING (Stabilizing)",
                                (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2, cv.LINE_AA)
                        else:
                            state = 0 # TRACKING
                            cv.putText(display, "STATE: TRACKING (Active)",
                                (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv.LINE_AA)

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

                        emit_eye_position(eye_mid, state, time.time())

                        # camera -> screen transform
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

        cv.imshow('Eye 3D Position (RGB-D / D455)', display)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\n[Tracker] Starting Screen Calibration. Stopping tracking stream...")
            pipeline.stop()
            cv.destroyAllWindows()
            calib_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrate_screen.py")
            os.system(f"python3 {calib_script}")
            import sys
            sys.exit(0)

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
    'iris_center_source': 'MediaPipe dedicated iris-center landmarks (468/473)',
    'camera_intrinsics': {
        'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy),
        'model': str(color_intr.model), 'coeffs': list(color_intr.coeffs)
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