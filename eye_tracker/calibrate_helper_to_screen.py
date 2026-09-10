"""
Helper (D435) -> Screen extrinsic calibration.

Step 2 of your slide deck: "Image Checkerboard targets on the screen with the
helper camera; solve T_helper->screen directly."

Prerequisites:
    1. screen_board_FINAL.png displayed FULLSCREEN on the monitor at 1:1 scale
       (use display_screen_board.py). Confirm with a ruler: squares = 25mm.
    2. Helper camera (D435) mounted off to the side, able to see the screen
       clearly and steadily.
    3. Helper camera RGB intrinsics. This script first tries to load a
       calibrated file (helper_intrinsics.npz with 'camera_matrix' and
       'dist_coeffs' keys, same format as your camera_intrinsics_printed.npz).
       If not found, it falls back to the D435's factory intrinsics reported
       by the RealSense SDK -- fine for a first pass, but a real checkerboard
       intrinsic calibration of the D435 RGB sensor will be more accurate.
    4. Both cameras may be plugged in -- this script explicitly selects the
       D435 by name so it doesn't accidentally grab the D455 instead.

Controls (during live view):
    c       -> capture current pose as one sample (only works when board is
               detected with enough corners)
    q / ESC -> stop capturing and save the averaged result

Output:
    helper_to_screen_calib.json containing:
        - R (3x3), t (3x1)         -> T_helper->screen as [R|t]
        - screen_center_helper_cam -> the (x,y,z) point in helper-cam frame
        - n_samples, translation_std_mm -> quality indicators
        - per_sample_reproj_errors_px, mean_reproj_error_px,
          max_reproj_error_px -> per-sample solvePnP reprojection error, so
          you can see how well each captured pose actually fits the detected
          corners (aim for well under 1 px; large outliers mean that sample's
          detection was noisy and you may want to recapture / average without it)
"""

import cv2
import numpy as np
import json
import os
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as Rot

# CONFIG

# Must match screen_board_FINAL.png / board_specs.json exactly
SCREEN_DICT       = cv2.aruco.DICT_4X4_50
SCREEN_SQUARES_X  = 10
SCREEN_SQUARES_Y  = 6
SQUARE_LEN_M      = 0.025
MARKER_LEN_M      = 0.01875

# Fixed offset from the board's true object-space origin (0,0,0) -- which is
# the OUTER corner of the board, per cv2.aruco.CharucoBoard's convention and
# what solvePnP's rvec/tvec are expressed relative to -- to the board's
# (=screen's) visual center, in the board's own flat plane.
#
# This is simply half the board's full physical size:
#   x: SCREEN_SQUARES_X * SQUARE_LEN_M / 2 = 10 * 0.025 / 2 = 0.125 m
#   y: SCREEN_SQUARES_Y * SQUARE_LEN_M / 2 =  6 * 0.025 / 2 = 0.075 m
#
# NOTE: do NOT measure this from the first *inner* ChArUco corner (id 0) --
# that corner sits one full square length inside the true origin, at
# (0.025, 0.025), and using it as the reference undershoots the true center
# by exactly one square length (25mm) in both x and y. Verified empirically
# by rendering the board, running it through detectBoard/matchImagePoints/
# solvePnP, and projecting (0,0,0) back onto the image -- it lands exactly
# on the outer top-left pixel of the board, not on the first inner corner.
CENTER_OFFSET_M = np.array([
    SCREEN_SQUARES_X * SQUARE_LEN_M / 2,
    SCREEN_SQUARES_Y * SQUARE_LEN_M / 2,
    0.0
])

HELPER_INTRINSICS_FILE = "helper_calib_intrinsics.npz"   # optional, see docstring
OUTPUT_FILE = "helper_to_screen_calib.json"

MIN_CORNERS_TO_ACCEPT = 8       # out of 45 total inner corners on this board
N_SAMPLES_TARGET      = 15      # stop automatically once this many are captured

STREAM_W, STREAM_H, FPS = 640, 480, 30   # matched to MAIN_RES/MAIN_FPS -- keep in sync with
                                          # STREAM_W/H in helper_intrinsic_calib.py, HELPER_RES in
                                          # main_to_helper_calib.py, and HELPER_RES in check_full_rig.py
WARMUP_FRAMES = 30  # discard frames right after start() while auto-exposure/WB settle



def find_device_serial(name_substring):
    ctx = rs.context()
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        if name_substring.lower() in name.lower():
            return dev.get_info(rs.camera_info.serial_number), name
    return None, None


def get_helper_intrinsics(profile):
    """Load a calibrated intrinsics file if present, else fall back to RealSense factory values."""
    if os.path.exists(HELPER_INTRINSICS_FILE):
        data = np.load(HELPER_INTRINSICS_FILE)
        print(f"Loaded calibrated intrinsics from {HELPER_INTRINSICS_FILE}")
        return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)

    print(f"WARNING: {HELPER_INTRINSICS_FILE} not found -- using D435 factory intrinsics. "
          f"Fine for a first pass, but a proper checkerboard intrinsic calibration "
          f"of the D435 RGB sensor will reduce error.")
    stream = profile.get_stream(rs.stream.color)
    intr = stream.as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    return K, dist


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(SCREEN_DICT)
    board = cv2.aruco.CharucoBoard((SCREEN_SQUARES_X, SCREEN_SQUARES_Y),
                                    SQUARE_LEN_M, MARKER_LEN_M, aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    helper_serial, helper_name = find_device_serial("D435")
    if helper_serial is None:
        print("ERROR: no D435 found. Check it's connected.")
        return
    print(f"Helper camera: {helper_name}  (serial {helper_serial})")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(helper_serial)
    config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    K, dist = get_helper_intrinsics(profile)

    print(f"Warming up auto-exposure/white-balance ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()

    rvecs_collected = []
    tvecs_collected = []
    reproj_errors_collected = []   # per-sample solvePnP reprojection RMS error, px

    print("\nLive view starting. Point the helper camera at the displayed screen board.")
    print(f"Press 'c' to capture a sample once the board is detected (need >= {MIN_CORNERS_TO_ACCEPT} corners).")
    print(f"Press 'q' or ESC to stop early and save. Auto-stops at {N_SAMPLES_TARGET} samples.\n")

    try:
        while len(rvecs_collected) < N_SAMPLES_TARGET:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())

            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

            # Normalize shapes -- OpenCV 5.0's detectBoard can return squeezed/
            # inconsistent shapes that make .total() disagree between corners
            # and ids even when len() matches, which trips an assertion in
            # drawDetectedCornersCharuco (and can also confuse matchImagePoints/
            # solvePnP downstream). Pure container reshape, no data change.
            if charuco_corners is not None:
                charuco_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
            if charuco_ids is not None:
                charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)

            vis = img.copy()
            corners_found = charuco_ids is not None and len(charuco_ids) >= MIN_CORNERS_TO_ACCEPT
            pnp_ok = False
            rvec, tvec = None, None
            reproj_err = None
            obj_points, img_points = None, None

            if corners_found:
                obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                # Normalize shapes/dtype -- matchImagePoints can return squeezed
                # (N, 2)/(N, 3) arrays instead of (N, 1, 2)/(N, 1, 3), which can
                # cause type mismatches downstream (e.g. in cv2.norm when computing
                # reprojection error). Pure container reshape, no data change.
                obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 1, 3)
                img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)
                pnp_ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)

                if pnp_ok:
                    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
                    reproj_err = cv2.norm(img_points, proj.reshape(-1, 1, 2), cv2.NORM_L2) / np.sqrt(len(proj))

            valid = corners_found and pnp_ok

            if valid:
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 0.05)
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                n = len(charuco_ids)
                cv2.putText(vis, f"corners: {n}  reproj: {reproj_err:.3f}px  [c]=capture  samples: {len(rvecs_collected)}/{N_SAMPLES_TARGET}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            elif corners_found and not pnp_ok:
                n = len(charuco_ids)
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                cv2.putText(vis, f"corners OK ({n}) but solvePnP FAILED -- check intrinsics",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, f"board not detected  samples: {len(rvecs_collected)}/{N_SAMPLES_TARGET}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("helper_to_screen_calib", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and valid:
                rvecs_collected.append(rvec)
                tvecs_collected.append(tvec)
                reproj_errors_collected.append(float(reproj_err))
                print(f"  captured sample {len(rvecs_collected)}/{N_SAMPLES_TARGET}  (reproj error: {reproj_err:.3f} px)")
            elif key in (ord('q'), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(rvecs_collected) == 0:
        print("No samples captured. Nothing saved.")
        return

    # --- average rotations properly (rotation vectors do NOT average linearly) ---
    rot_matrices = [cv2.Rodrigues(r)[0] for r in rvecs_collected]
    quats = Rot.from_matrix(np.array(rot_matrices)).as_quat()
    mean_rot = Rot.from_quat(quats).mean()
    R_avg = mean_rot.as_matrix()

    t_array = np.array([t.flatten() for t in tvecs_collected])
    t_avg = t_array.mean(axis=0)
    t_std_mm = t_array.std(axis=0) * 1000.0

    screen_center_helper_cam = R_avg @ CENTER_OFFSET_M + t_avg

    reproj_errors_arr = np.array(reproj_errors_collected)
    mean_reproj_err = float(reproj_errors_arr.mean())
    max_reproj_err = float(reproj_errors_arr.max())

    result = {
        "R": R_avg.tolist(),
        "t": t_avg.tolist(),
        "screen_center_helper_cam": screen_center_helper_cam.tolist(),
        "n_samples": len(rvecs_collected),
        "translation_std_mm": t_std_mm.tolist(),
        "per_sample_reproj_errors_px": reproj_errors_collected,
        "mean_reproj_error_px": mean_reproj_err,
        "max_reproj_error_px": max_reproj_err,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {OUTPUT_FILE}")
    print(f"n_samples = {len(rvecs_collected)}")
    print(f"translation std across samples (mm): {t_std_mm}  <- should be small (a few mm); large values mean unstable capture")
    print(f"per-sample reprojection error (px): {[f'{e:.3f}' for e in reproj_errors_collected]}")
    print(f"mean reprojection error (px): {mean_reproj_err:.4f}")
    print(f"max reprojection error (px): {max_reproj_err:.4f}")
    print(f"screen_center_helper_cam (meters): {screen_center_helper_cam}")
    print(f"\nAim for well under 1 px reprojection error per sample. Large outliers "
          f"usually mean that particular capture had a noisy/partial detection --consider "
          f"recapturing those poses or checking helper camera intrinsics if errors are "
          f"consistently high across all samples.")


if __name__ == "__main__":
    main()