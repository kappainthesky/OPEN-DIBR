#!/usr/bin/env python3

import sys
import os
import time
import json
import socket
import argparse
import threading
import numpy as np
import cv2 as cv
import mediapipe as mp
import pyrealsense2 as rs
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE_DEFAULT = os.path.join(SCRIPT_DIR, "main_to_screen_calib.json")
INTRINSICS_FILE_DEFAULT = os.path.join(SCRIPT_DIR, "main_calib_intrinsics.npz")

# Ring landmarks (4 pts each), used for overlay radius
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# MediaPipe dedicated iris-center landmarks (single point each)
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

# D455 rated depth range (mm)
MIN_VALID_DEPTH_MM = 200.0
MAX_VALID_DEPTH_MM = 4500.0
TARGET_SERIAL = "215122252978"

def parse_args():
    parser = argparse.ArgumentParser(description="Single-Camera D455 Screen-Calibrated Eye Tracking for OpenDIBR")
    parser.add_argument("--bridge", action="store_true", help="Connect to OpenDIBR TCP RGB-D frame server")
    parser.add_argument("--bridge-port", type=int, default=9998, help="OpenDIBR TCP frame server port (default: 9998)")
    parser.add_argument("--udp-ip", type=str, default="127.0.0.1", help="OpenDIBR UDP receiver IP (default: 127.0.0.1)")
    parser.add_argument("--udp-port", type=int, default=9999, help="OpenDIBR UDP receiver port (default: 9999)")
    parser.add_argument("--calib", type=str, default=CALIB_FILE_DEFAULT, help="Path to main_to_screen_calib.json")
    parser.add_argument("--intrinsics", type=str, default=INTRINSICS_FILE_DEFAULT, help="Path to main_calib_intrinsics.npz")
    parser.add_argument("--serial", type=str, default=None, help="RealSense device serial number to open")

    # Stabilizer Tuning Parameters (Tuned for ultra-low latency & zero jitter)
    parser.add_argument("--deadzone-x", type=float, default=3.0, help="X deadzone in mm (default: 3.0)")
    parser.add_argument("--deadzone-y", type=float, default=3.0, help="Y deadzone in mm (default: 3.0)")
    parser.add_argument("--deadzone-z", type=float, default=6.0, help="Z deadzone in mm (default: 6.0)")
    parser.add_argument("--alpha-still", type=float, default=0.40, help="Smoothing alpha when stationary (default: 0.40)")
    parser.add_argument("--alpha-move", type=float, default=0.85, help="Smoothing alpha when moving (default: 0.85)")
    parser.add_argument("--vel-threshold", type=float, default=20.0, help="Velocity threshold for motion in mm/s (default: 20.0)")
    parser.add_argument("--max-jump", type=float, default=140.0, help="Outlier jump rejection in mm (default: 140.0)")
    parser.add_argument("--gui", action="store_true", help="Show OpenCV debug window (default: False, headless)")
    parser.add_argument("--debug-log", action="store_true", help="Print stabilizer debug telemetry")
    return parser.parse_known_args()[0]


def check_realsense_device(target_serial=None):
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("\n[EyeTracker Error] No RealSense device detected.")
        sys.exit(1)

    found = None
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        if target_serial is not None and serial == target_serial:
            found = dev
        print(f"[EyeTracker] Detected: ... S/N: {serial} ...")

    if target_serial is not None:
        if found is None:
            print(f"[EyeTracker Error] Requested serial '{target_serial}' not found. "
                  f"Connected device(s): {[d.get_info(rs.camera_info.serial_number) for d in devices]}")
            sys.exit(1)
        return found
    return devices[0]


class LatestFrameReceiver(threading.Thread):
    """
    Asynchronous TCP frame receiver that continuously reads RGB-D frames
    from OpenDIBR into a shared buffer, immediately overwriting older frames.
    Auto-reconnects if connection is dropped.
    """
    def __init__(self, host, port, frame_bytes):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.frame_bytes = frame_bytes
        self.sock = None
        self.running = True
        self.connected = False
        self.lock = threading.Lock()
        self.has_frame = threading.Event()
        self.latest_buf = None
        self.latest_time = 0.0

    def connect(self):
        while self.running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((self.host, self.port))
                self.sock = s
                self.connected = True
                print(f"[EyeTracker] Successfully connected to OpenDIBR TCP RGB-D bridge ({self.host}:{self.port})!")
                return True
            except Exception:
                time.sleep(0.3)
        return False

    def run(self):
        buf = bytearray(self.frame_bytes)
        view = memoryview(buf)
        while self.running:
            if not self.connected or self.sock is None:
                if not self.connect():
                    break

            n = 0
            while n < self.frame_bytes and self.running:
                try:
                    chunk = self.sock.recv_into(view[n:])
                    if chunk == 0:
                        self.connected = False
                        break
                    n += chunk
                except Exception:
                    self.connected = False
                    break

            if n == self.frame_bytes:
                t_now = time.perf_counter()
                with self.lock:
                    self.latest_buf = bytes(buf)
                    self.latest_time = t_now
                self.has_frame.set()
            elif not self.connected:
                if self.sock:
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                    self.sock = None
                time.sleep(0.2)

    def get_latest_frame(self, timeout=0.1):
        if not self.has_frame.wait(timeout=timeout):
            return None, 0.0
        with self.lock:
            self.has_frame.clear()
            return self.latest_buf, self.latest_time

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


class EyeStabilizer:
    def __init__(self,
                 deadzone_x=3.0,
                 deadzone_y=3.0,
                 deadzone_z=6.0,
                 alpha_still=0.40,
                 alpha_move=0.85,
                 vel_threshold=20.0,
                 max_jump_mm=140.0):
        self.deadzone = np.array([deadzone_x, deadzone_y, deadzone_z], dtype=np.float32)
        self.alpha_still = alpha_still
        self.alpha_move = alpha_move
        self.vel_threshold = vel_threshold
        self.max_jump_mm = max_jump_mm

        self.stable_pos = None      # Output position sent to OpenDIBR
        self.filtered_pos = None    # Continuous filtered position
        self.last_raw = None        # Previous raw position for jump detection
        self.last_time = None
        self.outlier_count = 0

    def update(self, raw_pos, timestamp=None):
        if timestamp is None:
            timestamp = time.perf_counter()
        raw = np.array(raw_pos, dtype=np.float32)

        # Bootstrap
        if self.stable_pos is None:
            self.stable_pos = raw.copy()
            self.filtered_pos = raw.copy()
            self.last_raw = raw.copy()
            self.last_time = timestamp
            return self.stable_pos.copy(), "INIT", True

        dt = max(1e-3, timestamp - self.last_time)
        self.last_time = timestamp

        # Outlier Rejection
        jump_dist = float(np.linalg.norm(raw - self.last_raw))
        if jump_dist > self.max_jump_mm and self.outlier_count < 2:
            self.outlier_count += 1
            self.last_raw = raw.copy()  # bugfix: keep last_raw current even when rejecting
            return self.stable_pos.copy(), f"OUTLIER_REJECTED (jump={jump_dist:.1f}mm)", False
        self.outlier_count = 0
        self.last_raw = raw.copy()

        # Adaptive Velocity-Based EMA Filter
        inst_vel = jump_dist / dt
        vel_factor = float(np.clip(inst_vel / max(1.0, self.vel_threshold), 0.0, 1.0))
        alpha_eff = self.alpha_still + (self.alpha_move - self.alpha_still) * (vel_factor ** 1.2)
        self.filtered_pos = alpha_eff * raw + (1.0 - alpha_eff) * self.filtered_pos

        # 3-Axis Independent Deadzone
        delta = self.filtered_pos - self.stable_pos
        abs_delta = np.abs(delta)
        exceeded = abs_delta > self.deadzone

        if np.any(exceeded):
            self.stable_pos = self.filtered_pos.copy()
            status = f"MOVING (d=[{delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}]mm)"
            accepted = True
        else:
            status = f"STABLE_HELD (d=[{delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}]mm < dz)"
            accepted = False

        return self.stable_pos.copy(), status, accepted


def load_screen_calibration(path):
    """Load the main-camera -> screen-center calibration.
    
    Transforms 3D points from native RealSense camera frame into screen coordinates
    where (0, 0) is the exact physical monitor center.
    """
    try:
        with open(path, 'r') as f:
            calib = json.load(f)

        R = np.array(calib['R'], dtype=np.float32)
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {R.shape}")

        screen_center_m = np.array(calib['screen_center_main_cam'], dtype=np.float32)
        screen_center_mm = screen_center_m * 1000.0

        dist_mm = calib.get('distance_main_to_screen_center_mm', float(np.linalg.norm(screen_center_mm)))
        print(f"[Screen Calibration] Loaded from '{os.path.basename(path)}'")
        print(f" -> Screen Center in D455 Frame: x={screen_center_mm[0]:+.1f}, y={screen_center_mm[1]:+.1f}, z={screen_center_mm[2]:.1f} mm")
        print(f" -> Main-to-Screen Distance   : {dist_mm:.1f} mm")
        return R, screen_center_mm, True

    except Exception as e:
        print(f"\n[Screen Calibration Warning] Could not load '{path}': {e}")
        print(" -> Using identity screen transform.\n")
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), False


def load_camera_intrinsics(npz_path):
    """Loads calibrated camera matrix and distortion coefficients.

    Uses calibrated intrinsics (main_calib_intrinsics.npz) rather than
    RealSense factory intrinsics, since main_to_screen_calib.json was
    itself solved using these calibrated intrinsics -- the runtime
    deprojection here needs to match that, or the screen-center
    transform won't line up correctly (see the intrinsics/extrinsic
    calibration consistency discussion).
    """
    intr = rs.intrinsics()
    intr.width = 640
    intr.height = 480
    intr.ppx = 325.05
    intr.ppy = 239.86
    intr.fx = 383.00
    intr.fy = 382.55
    intr.model = rs.distortion.brown_conrady
    intr.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            K = data['camera_matrix']
            dist = data['dist_coeffs'].flatten()
            intr.fx = float(K[0, 0])
            intr.fy = float(K[1, 1])
            intr.ppx = float(K[0, 2])
            intr.ppy = float(K[1, 2])
            intr.coeffs = [float(c) for c in dist[:5]]
            print(f"[Intrinsics] Loaded from '{os.path.basename(npz_path)}': fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")
        except Exception as e:
            print(f"[Intrinsics Warning] Could not load '{npz_path}': {e}")
    else:
        print(f"[Intrinsics Warning] '{npz_path}' not found -- using D455 factory intrinsics fallback. "
              f"This will not match main_to_screen_calib.json if that was solved with calibrated intrinsics.")

    return intr


def get_aligned_depth(depth_img, u, v, patch=3):
    """Median depth in a small symmetric patch around (u, v) in the aligned depth frame."""
    if depth_img is None:
        return 0.0
    u, v = int(u), int(v)
    h, w = depth_img.shape
    u = max(patch, min(w - patch - 1, u))
    v = max(patch, min(h - patch - 1, v))
    region = depth_img[v - patch:v + patch + 1, u - patch:u + patch + 1]
    valid = region[region > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def pixel_to_3d(u, v, depth_mm, color_intr, width=640, mirror=False):
    """Deprojects a pixel in the flipped (mirror) frame to a 3D point."""
    u_native = (width - 1) - u  # undo horizontal flip for camera frame

    point = rs.rs2_deproject_pixel_to_point(
        color_intr,
        [float(u_native), float(v)],
        float(depth_mm)
    )
    point = np.array(point, dtype=np.float32)
    if mirror:
        point[0] = -point[0]
    return point


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print(" [OpenDIBR Screen-Calibrated Eye Tracking (Intel RealSense D455)]")
    print("=" * 65)

    SCREEN_R, SCREEN_CENTER_MM, SCREEN_CALIBRATED = load_screen_calibration(args.calib)

    if SCREEN_CALIBRATED:
        print("[Screen Calibration] ENABLED — using calibrated screen-center transform.")
    else:
        print("[Screen Calibration] WARNING — calib file failed to load; falling back to identity transform (raw camera-frame coordinates).")

    def camera_to_screen(p_cam):
        return SCREEN_R.T @ (p_cam - SCREEN_CENTER_MM)

    # Intrinsics setup (calibrated, matching main_to_screen_calib.json)
    color_intr = load_camera_intrinsics(args.intrinsics)

    # UDP socket for sending coordinates to OpenDIBR C++ application
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[EyeTracker] Sending UDP eye packets to {args.udp_ip}:{args.udp_port}")

    # Stabilizer
    stabilizer = EyeStabilizer(
        deadzone_x=args.deadzone_x,
        deadzone_y=args.deadzone_y,
        deadzone_z=args.deadzone_z,
        alpha_still=args.alpha_still,
        alpha_move=args.alpha_move,
        vel_threshold=args.vel_threshold,
        max_jump_mm=args.max_jump
    )

    # Video Source Setup (Bridge or Standalone)
    frame_receiver = None
    pipeline = None
    align = None

    frame_bytes = 640 * 480 * 3 + 640 * 480 * 2
    rgb_bytes = 640 * 480 * 3

    if args.bridge:
        print(f"[EyeTracker] Bridge mode active: Connecting to OpenDIBR TCP server on 127.0.0.1:{args.bridge_port}...")
        frame_receiver = LatestFrameReceiver("127.0.0.1", args.bridge_port, frame_bytes)
        frame_receiver.start()
        print("[EyeTracker] Asynchronous auto-reconnecting frame receiver thread started.")
    else:
        print("[EyeTracker] Standalone mode: Checking for connected RealSense device...")
        check_realsense_device(target_serial=TARGET_SERIAL)

        print("[EyeTracker] Standalone mode: Initializing RealSense D455 pipeline directly...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(TARGET_SERIAL) 
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            profile = pipeline.start(config)
        except Exception as e:
            print(f"\n[EyeTracker Error] Failed to start RealSense pipeline: {e}")
            print(" -> The device was detected but could not be opened — it may be held by another")
            print("    process (RealSense Viewer, a previous crashed run), or the requested stream")
            print("    format/resolution/FPS combo may not be supported by this firmware.")
            sys.exit(1)
        align = rs.align(rs.stream.color)
        print(f"[EyeTracker] Direct RealSense D455 pipeline running at 640x480 30FPS.")
        print(f"[EyeTracker] Standalone mode still uses the calibrated intrinsics loaded above "
              f"(not the live device's factory intrinsics), to stay consistent with main_to_screen_calib.json.")

    print(f"[EyeTracker] Deadzone: X={args.deadzone_x}mm, Y={args.deadzone_y}mm, Z={args.deadzone_z}mm")
    print(f"[EyeTracker] Smoothing: Still Alpha={args.alpha_still}, Move Alpha={args.alpha_move}")
    print("=" * 65 + "\n")

    last_log_time = time.perf_counter()
    last_accepted_state = None
    frame_idx = 0
    fps_start_time = time.perf_counter()
    fps_counter = 0
    current_track_fps = 30.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as landmarker:
        print("Eye tracker running — press 'q' to quit.\n")

        try:
            while True:
                # Capture frame
                if args.bridge:
                    raw_buf, t_capture = frame_receiver.get_latest_frame(timeout=0.1)
                    if raw_buf is None:
                        if not frame_receiver.running:
                            print("[EyeTracker Bridge] Connection closed by OpenDIBR. Exiting.")
                            break
                        continue

                    # OpenDIBR pushes RGB8 frames over bridge
                    color_img_rgb = np.frombuffer(raw_buf[:rgb_bytes], dtype=np.uint8).reshape((480, 640, 3))
                    depth_img = np.frombuffer(raw_buf[rgb_bytes:frame_bytes], dtype=np.uint16).reshape((480, 640))
                else:
                    frames = pipeline.wait_for_frames()
                    t_capture = time.perf_counter()
                    aligned = align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    color_img_rgb = np.asanyarray(color_frame.get_data())
                    depth_img = np.asanyarray(depth_frame.get_data())

                # Measure MediaPipe processing latency
                t_mp_start = time.perf_counter()

                # Flip horizontally for mirror view alignment
                frame_rgb = cv.flip(color_img_rgb, 1)
                depth_flp = np.fliplr(depth_img)

                img_h, img_w = frame_rgb.shape[:2]

                # FaceMesh expects RGB format
                results = landmarker.process(frame_rgb)

                t_mp_end = time.perf_counter()
                t_mp_ms = (t_mp_end - t_mp_start) * 1000.0

                fps_counter += 1
                if t_mp_end - fps_start_time >= 1.0:
                    current_track_fps = fps_counter / (t_mp_end - fps_start_time)
                    fps_counter = 0
                    fps_start_time = t_mp_end

                cur_time = t_mp_end

                # Prepare BGR display for OpenCV GUI if needed
                display = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR) if args.gui else None

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    mesh_points = np.array([
                        np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                        for p in landmarks
                    ])

                    l_u, l_v = mesh_points[LEFT_IRIS_CENTER]
                    r_u, r_v = mesh_points[RIGHT_IRIS_CENTER]

                    _, l_r = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    _, r_r = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                    # Aligned depth sampling
                    l_depth_raw = get_aligned_depth(depth_flp, l_u, l_v)
                    r_depth_raw = get_aligned_depth(depth_flp, r_u, r_v)

                    valid_depths = [d for d in (l_depth_raw, r_depth_raw)
                                     if MIN_VALID_DEPTH_MM <= d <= MAX_VALID_DEPTH_MM]

                    if len(valid_depths) == 0:
                        # Both eyes failed sensor depth -- skip 3D projection
                        # this frame (original behavior), report as LOST.
                        state = 2  # LOST
                        accepted = False
                        status_msg = "NO_VALID_DEPTH"
                        eye_mid_screen = np.array([0.0, 0.0, 650.0], dtype=np.float32)
                        p_stabilized = stabilizer.stable_pos if stabilizer.stable_pos is not None else eye_mid_screen

                        msg = (f"{p_stabilized[0]:.2f} {p_stabilized[1]:.2f} {p_stabilized[2]:.2f} {state} "
                               f"0.0 0.0 0.0 2 0.0 0.0 0.0 2 "
                               f"{t_capture:.4f} {t_mp_ms:.1f} {current_track_fps:.1f}")
                        try:
                            udp_sock.sendto(msg.encode("utf-8"), (args.udp_ip, args.udp_port))
                        except Exception:
                            pass

                        if args.gui:
                            cv.putText(display, "No valid depth (too close / out of range)",
                                       (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)
                    else:
                        l_depth = l_depth_raw if l_depth_raw > 0 else np.median(valid_depths)
                        r_depth = r_depth_raw if r_depth_raw > 0 else np.median(valid_depths)

                        # Native 3D projection in D455 camera frame
                        l_3d_native = pixel_to_3d(l_u, l_v, l_depth, color_intr, width=img_w, mirror=False)
                        r_3d_native = pixel_to_3d(r_u, r_v, r_depth, color_intr, width=img_w, mirror=False)
                        eye_mid_native = (l_3d_native + r_3d_native) / 2.0

                        # Screen-relative transformation (Screen Center = (0, 0, 0))
                        eye_mid_screen = camera_to_screen(eye_mid_native)
                        l_screen = camera_to_screen(l_3d_native)
                        r_screen = camera_to_screen(r_3d_native)

                        # Multi-stage stabilization
                        p_stabilized, status_msg, accepted = stabilizer.update(eye_mid_screen, cur_time)
                        state = 0  # TRACKING

                        # Transmit UDP message with timestamp & performance telemetry to OpenDIBR
                        msg = (f"{p_stabilized[0]:.2f} {p_stabilized[1]:.2f} {p_stabilized[2]:.2f} {state} "
                               f"{l_screen[0]:.2f} {l_screen[1]:.2f} {l_screen[2]:.2f} 0 "
                               f"{r_screen[0]:.2f} {r_screen[1]:.2f} {r_screen[2]:.2f} 0 "
                               f"{t_capture:.4f} {t_mp_ms:.1f} {current_track_fps:.1f}")
                        try:
                            udp_sock.sendto(msg.encode("utf-8"), (args.udp_ip, args.udp_port))
                        except Exception:
                            pass

                        # GUI Overlay
                        if args.gui:
                            center_left = np.array([l_u, l_v], dtype=np.int32)
                            center_right = np.array([r_u, r_v], dtype=np.int32)

                            cv.circle(display, center_left, int(l_r), (255, 0, 0), 2, cv.LINE_AA)
                            cv.circle(display, center_right, int(r_r), (255, 0, 0), 2, cv.LINE_AA)
                            cv.circle(display, center_left, 2, (0, 0, 255), -1, cv.LINE_AA)
                            cv.circle(display, center_right, 2, (0, 0, 255), -1, cv.LINE_AA)

                            cv.putText(display, f"SCREEN: x={p_stabilized[0]:+.1f} y={p_stabilized[1]:+.1f} z={p_stabilized[2]:.1f} mm",
                                       (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
                            cv.putText(display, f"Raw Screen: x={eye_mid_screen[0]:+.1f} y={eye_mid_screen[1]:+.1f} z={eye_mid_screen[2]:.1f} mm",
                                       (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv.LINE_AA)
                            cv.putText(display, f"Status: {status_msg} | MP: {t_mp_ms:.1f}ms | FPS: {current_track_fps:.1f}",
                                       (10, 95), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)
                else:
                    # Face lost
                    state = 2  # LOST
                    accepted = False
                    status_msg = "FACE_LOST"
                    eye_mid_screen = np.array([0.0, 0.0, 650.0], dtype=np.float32)
                    p_stabilized = stabilizer.stable_pos if stabilizer.stable_pos is not None else eye_mid_screen
                    msg = (f"{p_stabilized[0]:.2f} {p_stabilized[1]:.2f} {p_stabilized[2]:.2f} {state} "
                           f"0.0 0.0 0.0 2 0.0 0.0 0.0 2 "
                           f"{t_capture:.4f} {t_mp_ms:.1f} {current_track_fps:.1f}")
                    try:
                        udp_sock.sendto(msg.encode("utf-8"), (args.udp_ip, args.udp_port))
                    except Exception:
                        pass

                    if args.gui:
                        cv.putText(display, "Face Lost / Searching...", (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)

                # Debug Telemetry
                if args.debug_log and (cur_time - last_log_time >= 0.5 or accepted != last_accepted_state):
                    last_log_time = cur_time
                    last_accepted_state = accepted
                    if results.multi_face_landmarks:
                        tag = "[ACCEPTED]" if accepted else "[HELD]"
                        print(f"[Tracker] Raw Screen=({eye_mid_screen[0]:+6.1f}, {eye_mid_screen[1]:+6.1f}, {eye_mid_screen[2]:6.1f}) mm | "
                              f"Filt=({p_stabilized[0]:+6.1f}, {p_stabilized[1]:+6.1f}, {p_stabilized[2]:6.1f}) mm {tag} {status_msg} | MP: {t_mp_ms:.1f}ms")

                if args.gui:
                    cv.imshow("OpenDIBR Screen-Calibrated Eye Tracker", display)
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n[EyeTracker] Stopped by user.")
        finally:
            if frame_receiver is not None:
                frame_receiver.stop()
            if pipeline is not None:
                pipeline.stop()
            if args.gui:
                cv.destroyAllWindows()

    print(f"[EyeTracker] Total frames processed: {frame_idx}. Exiting cleanly.")


if __name__ == "__main__":
    main()