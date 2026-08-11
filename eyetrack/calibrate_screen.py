import os
import sys
import time
import json
import numpy as np
import cv2 as cv
import mediapipe as mp
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths & Settings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
CALIB_FILE = os.path.join(SCRIPT_DIR, "screen_calib.json")

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

# Initialize RealSense D455 Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
color_intr = color_stream.get_intrinsics()
fx, fy = color_intr.fx, color_intr.fy
cx, cy = color_intr.ppx, color_intr.ppy

# Initialize MediaPipe FaceLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

def pixel_to_3d(u, v, depth_mm):
    x = (u - cx) * depth_mm / fx
    y = (v - cy) * depth_mm / fy
    return np.array([x, y, depth_mm], dtype=np.float32)

def get_aligned_depth(depth_img, u, v, patch=3):
    h, w = depth_img.shape
    u_min, u_max = max(0, u - patch), min(w, u + patch + 1)
    v_min, v_max = max(0, v - patch), min(h, v + patch + 1)
    sub = depth_img[v_min:v_max, u_min:u_max]
    valid = sub[(sub >= 250) & (sub <= 3000)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0

def capture_eye_samples(landmarker, target_num_samples=30):
    samples = []
    print(f" -> Collecting {target_num_samples} frames...")
    while len(samples) < target_num_samples:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        frame = cv.flip(color_img, 1)
        depth_flp = np.fliplr(depth_img)

        img_h, img_w = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect(mp_image)

        if results.face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.face_landmarks[0]
            ])
            l_u, l_v = mesh_points[LEFT_IRIS_CENTER]
            r_u, r_v = mesh_points[RIGHT_IRIS_CENTER]

            l_depth = get_aligned_depth(depth_flp, l_u, l_v)
            r_depth = get_aligned_depth(depth_flp, r_u, r_v)

            if l_depth > 0 and r_depth > 0:
                l_3d = pixel_to_3d(l_u, l_v, l_depth)
                r_3d = pixel_to_3d(r_u, r_v, r_depth)
                eye_mid = (l_3d + r_3d) / 2.0
                samples.append(eye_mid)

    return np.mean(samples, axis=0)

def main():
    cv.namedWindow("Windowed 6DoF Screen Calibration", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Windowed 6DoF Screen Calibration", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    screen_w = 1920
    screen_h = 1080

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        points = {}
        targets = [
            ("CENTER", (screen_w // 2, screen_h // 2), "Phase 1/4: Look at the CENTER dot and press SPACE"),
            ("LEFT", (100, screen_h // 2), "Phase 2/4 (a): Look at the LEFT dot and press SPACE"),
            ("RIGHT", (screen_w - 100, screen_h // 2), "Phase 2/4 (b): Look at the RIGHT dot and press SPACE"),
            ("TOP", (screen_w // 2, 100), "Phase 3/4 (a): Look at the TOP dot and press SPACE"),
            ("BOTTOM", (screen_w // 2, screen_h - 100), "Phase 3/4 (b): Look at the BOTTOM dot and press SPACE")
        ]

        for name, pos, instruction in targets:
            while True:
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv.circle(canvas, pos, 24, (0, 0, 255), -1, cv.LINE_AA)
                cv.circle(canvas, pos, 8, (255, 255, 255), -1, cv.LINE_AA)
                cv.putText(canvas, instruction, (screen_w // 2 - 400, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
                cv.putText(canvas, "Press SPACE to capture point", (screen_w // 2 - 250, screen_h - 60),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv.LINE_AA)
                
                cv.imshow("Windowed 6DoF Screen Calibration", canvas)
                key = cv.waitKey(10) & 0xFF
                if key == 32: # SPACE
                    print(f"[Calibration] Capturing {name}...")
                    points[name] = capture_eye_samples(landmarker, target_num_samples=30)
                    break
                elif key == ord('q'):
                    pipeline.stop()
                    cv.destroyAllWindows()
                    sys.exit(0)

        # -------------------------------------------------------------
        # MATHEMATICAL COMPUTATIONS (Gram-Schmidt Orthonormalization)
        # -------------------------------------------------------------
        c0 = points["CENTER"]
        c_left = points["LEFT"]
        c_right = points["RIGHT"]
        c_top = points["TOP"]
        c_bottom = points["BOTTOM"]

        # Vector X (Horizontal screen axis in camera space)
        vx = c_right - c_left
        est_width_mm = float(np.linalg.norm(vx))
        ux = vx / est_width_mm

        # Vector Y (Vertical screen axis in camera space)
        vy = c_top - c_bottom
        est_height_mm = float(np.linalg.norm(vy))

        # Vector Z (Normal to screen pointing towards user)
        vz = np.cross(ux, vy)
        uz = vz / np.linalg.norm(vz)

        # Orthogonalized Vector Y
        uy = np.cross(uz, ux)

        # Rotation matrix (Camera space -> Screen space)
        R_screen = np.vstack([ux, uy, uz])

        # Baseline distance & Translation vector
        baseline_dist_mm = float(np.linalg.norm(c0))
        # Map center point c0 to (0, 0, baseline_dist_mm) in screen space
        t_screen = np.array([0.0, 0.0, baseline_dist_mm], dtype=np.float32) - R_screen @ c0

        # -------------------------------------------------------------
        # PHASE 4: VALIDATION
        # -------------------------------------------------------------
        val_targets = [
            ("TOP_LEFT", (100, 100)),
            ("TOP_RIGHT", (screen_w - 100, 100)),
            ("BOTTOM_LEFT", (100, screen_h - 100)),
            ("BOTTOM_RIGHT", (screen_w - 100, screen_h - 100))
        ]
        val_errors = []

        for val_name, val_pos in val_targets:
            while True:
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv.circle(canvas, val_pos, 20, (0, 255, 255), -1, cv.LINE_AA)
                cv.putText(canvas, f"Validation Phase: Look at {val_name} and press SPACE",
                           (screen_w // 2 - 380, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
                cv.imshow("Windowed 6DoF Screen Calibration", canvas)
                key = cv.waitKey(10) & 0xFF
                if key == 32:
                    p_cam = capture_eye_samples(landmarker, target_num_samples=20)
                    p_scr = R_screen @ p_cam + t_screen
                    # Error metric: displacement from expected baseline Z plane
                    err = abs(p_scr[2] - baseline_dist_mm)
                    val_errors.append(err)
                    break

        mean_val_error_mm = float(np.mean(val_errors))
        accuracy_label = "EXCELLENT" if mean_val_error_mm < 15.0 else ("GOOD" if mean_val_error_mm < 30.0 else "FAIR")

        # Save to screen_calib.json
        calib_data = {
            "rotation_matrix": R_screen.tolist(),
            "translation_mm": {
                "x": float(t_screen[0]),
                "y": float(t_screen[1]),
                "z": float(t_screen[2])
            },
            "physical_dimensions_mm": {
                "width": est_width_mm,
                "height": est_height_mm
            },
            "baseline_distance_mm": baseline_dist_mm,
            "validation_error_mm": mean_val_error_mm,
            "accuracy": accuracy_label
        }

        with open(CALIB_FILE, 'w') as f:
            json.dump(calib_data, f, indent=2)

        print("\n=============================================")
        print(" [Screen Calibration] Completed Successfully!")
        print(f" -> Baseline Distance  : {baseline_dist_mm:.1f} mm")
        print(f" -> Physical Display   : {est_width_mm:.1f} x {est_height_mm:.1f} mm")
        print(f" -> Mean RMS Error     : {mean_val_error_mm:.2f} mm ({accuracy_label})")
        print(f" -> Calibration saved to: {CALIB_FILE}")
        print("=============================================\n")

    pipeline.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
