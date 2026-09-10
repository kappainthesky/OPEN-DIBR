"""
Main (D455) <-> Helper (D435) bridging calibration via a shared printed board.

Step 1 of your slide deck: place the board once, capture its pose from the
main camera, then -- WITHOUT MOVING THE BOARD -- capture its pose from the
helper camera. Compute:

    T_main->helper = T_main->board * inv(T_helper->board)

Cameras are used SEQUENTIALLY, one pipeline open at a time -- no simultaneous
streaming, no USB bandwidth contention, both can be on the same laptop
regardless of which ports/USB versions they're on.

Prerequisites:
    1. print_board_main_helper.pdf printed at true scale (30mm squares,
       verified with a ruler) and mounted flat/rigid.
    2. camera_intrinsics_printed.npz -- your existing D455 intrinsics
       (must match MAIN_RES below).
    3. helper_intrinsics.npz -- D435 factory intrinsics you already saved
       (must match HELPER_RES below).
    4. Both cameras plugged in. The script auto-selects devices by name
       (looks for "D455" and "D435" in the product name).

Procedure:
    1. Place the board somewhere both cameras can eventually see it (it will
       stay in this exact spot for the whole process -- do not move it once
       step 2 starts).
    2. Script opens the MAIN (D455) camera. Aim it at the board, press 'c'
       to capture several samples, then 'q' to move on.
    3. Script opens the HELPER (D435) camera. Aim IT at the board (same
       board, same position -- do not touch the board itself), press 'c'
       to capture, 'q' to finish.
    4. Script computes and saves T_main->helper.

Output:
    main_to_helper_calib.json containing R (3x3), t (3x1), n_samples_main,
    n_samples_helper, translation_std_mm for both captures, and per-camera
    solvePnP reprojection error stats (mean_reproj_error_px_main/helper,
    max_reproj_error_px_main/helper) so you can check how well each set of
    captures actually fit the board before trusting the bridging result.
"""

import cv2
import numpy as np
import json
import os
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as Rot

# CONFIG

# Must match print_board_main_helper.png / board_specs.json exactly
PRINT_DICT      = cv2.aruco.DICT_5X5_1000
PRINT_SQUARES_X = 8
PRINT_SQUARES_Y = 6
SQUARE_LEN_M    = 0.030   # verify against your printed board with a ruler
MARKER_LEN_M    = 0.022

MAIN_INTRINSICS_FILE   = "main_calib_intrinsics_factory.npz"
HELPER_INTRINSICS_FILE = "helper_calib_intrinsics.npz"

MAIN_RES   = (640, 480)    # must match main_calib_intrinsics.npz
MAIN_FPS   = 30
HELPER_RES = (640, 480)    # matched to MAIN_RES -- must match helper_calib_intrinsics.npz;
                            # keep in sync with STREAM_W/H in helper_intrinsic_calib.py,
                            # STREAM_W/H in helper_to_screen_calib.py, and HELPER_RES in
                            # check_full_rig.py
HELPER_FPS = 30

OUTPUT_FILE = "main_to_helper_calib.json"

MIN_CORNERS_TO_ACCEPT = 8
N_SAMPLES_TARGET = 15



def find_device_serial(name_substring):
    ctx = rs.context()
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        if name_substring.lower() in name.lower():
            return dev.get_info(rs.camera_info.serial_number), name
    return None, None


def load_intrinsics(path):
    data = np.load(path)
    return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)


WARMUP_FRAMES = 30  # discard this many frames after pipeline.start() for auto-exposure/WB to settle


def capture_pose(serial, label, width, height, fps, K, dist, board, detector):
    """Open one camera, live-detect the board, let user capture N samples, return averaged R, t."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)

    print(f"\n=== {label} camera live view ===")
    print(f"Warming up auto-exposure/white-balance ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()

    rvecs_collected = []
    tvecs_collected = []
    reproj_errors_collected = []   # per-sample solvePnP reprojection RMS error, px

    print(f"Aim it at the board (do not move the board between cameras).")
    print(f"Press 'c' to capture (need >= {MIN_CORNERS_TO_ACCEPT} corners). "
          f"'q'/ESC to finish. Auto-stops at {N_SAMPLES_TARGET}.\n")

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
                cv2.putText(vis, f"[{label}] corners: {n}  reproj: {reproj_err:.3f}px  [c]=capture  {len(rvecs_collected)}/{N_SAMPLES_TARGET}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            elif corners_found and not pnp_ok:
                # corners were found fine -- solvePnP itself is the failure (bad K/dist is a common cause)
                n = len(charuco_ids)
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                cv2.putText(vis, f"[{label}] corners OK ({n}) but solvePnP FAILED -- check intrinsics",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, f"[{label}] no corners detected  {len(rvecs_collected)}/{N_SAMPLES_TARGET}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(f"{label}_capture", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and valid:
                rvecs_collected.append(rvec)
                tvecs_collected.append(tvec)
                reproj_errors_collected.append(float(reproj_err))
                print(f"  [{label}] captured {len(rvecs_collected)}/{N_SAMPLES_TARGET}  (reproj error: {reproj_err:.3f} px)")
            elif key in (ord('q'), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(rvecs_collected) == 0:
        return None, None, 0, None, None

    rot_matrices = [cv2.Rodrigues(r)[0] for r in rvecs_collected]
    quats = Rot.from_matrix(np.array(rot_matrices)).as_quat()
    R_avg = Rot.from_quat(quats).mean().as_matrix()

    t_array = np.array([t.flatten() for t in tvecs_collected])
    t_avg = t_array.mean(axis=0)
    t_std_mm = (t_array.std(axis=0) * 1000.0).tolist()

    reproj_stats = {
        "per_sample_reproj_errors_px": reproj_errors_collected,
        "mean_reproj_error_px": float(np.mean(reproj_errors_collected)),
        "max_reproj_error_px": float(np.max(reproj_errors_collected)),
    }

    return R_avg, t_avg, len(rvecs_collected), t_std_mm, reproj_stats


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(PRINT_DICT)
    board = cv2.aruco.CharucoBoard((PRINT_SQUARES_X, PRINT_SQUARES_Y),
                                    SQUARE_LEN_M, MARKER_LEN_M, aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    main_serial, main_name = find_device_serial("D455")
    helper_serial, helper_name = find_device_serial("D435")

    if main_serial is None:
        print("ERROR: no D455 found. Check it's connected.")
        return
    if helper_serial is None:
        print("ERROR: no D435 found. Check it's connected.")
        return

    print(f"Main camera:   {main_name}  (serial {main_serial})")
    print(f"Helper camera: {helper_name}  (serial {helper_serial})")

    K_main, dist_main = load_intrinsics(MAIN_INTRINSICS_FILE)
    K_helper, dist_helper = load_intrinsics(HELPER_INTRINSICS_FILE)

    input("\nPlace the board in its fixed position, then press Enter to start with the MAIN camera...")

    R_main, t_main, n_main, std_main, reproj_main = capture_pose(
        main_serial, "MAIN", MAIN_RES[0], MAIN_RES[1], MAIN_FPS, K_main, dist_main, board, detector)

    if R_main is None:
        print("No samples captured from main camera. Aborting.")
        return

    print(f"\nMain camera done: {n_main} samples, translation std (mm): {std_main}, "
          f"mean reproj error (px): {reproj_main['mean_reproj_error_px']:.4f}, "
          f"max reproj error (px): {reproj_main['max_reproj_error_px']:.4f}")
    input("\nDO NOT MOVE THE BOARD. Press Enter to continue with the HELPER camera...")

    R_helper, t_helper, n_helper, std_helper, reproj_helper = capture_pose(
        helper_serial, "HELPER", HELPER_RES[0], HELPER_RES[1], HELPER_FPS, K_helper, dist_helper, board, detector)

    if R_helper is None:
        print("No samples captured from helper camera. Aborting.")
        return

    print(f"\nHelper camera done: {n_helper} samples, translation std (mm): {std_helper}, "
          f"mean reproj error (px): {reproj_helper['mean_reproj_error_px']:.4f}, "
          f"max reproj error (px): {reproj_helper['max_reproj_error_px']:.4f}")

    # T_main->helper = T_main->board * inv(T_helper->board)
    R_mh = R_main @ R_helper.T
    t_mh = t_main - R_mh @ t_helper

    result = {
        "R": R_mh.tolist(),
        "t": t_mh.tolist(),
        "n_samples_main": n_main,
        "n_samples_helper": n_helper,
        "translation_std_mm_main": std_main,
        "translation_std_mm_helper": std_helper,
        "reproj_error_main": reproj_main,
        "reproj_error_helper": reproj_helper,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {OUTPUT_FILE}")
    print(f"T_main->helper rotation:\n{R_mh}")
    print(f"T_main->helper translation (m): {t_mh}")
    print(f"\nIf either translation_std is more than a few mm, recapture that "
          f"camera's samples -- likely an unstable hand-hold or the board "
          f"partially leaving frame during capture.")
    print(f"If either mean/max reprojection error is high (aim for well under 1 px), "
          f"that camera's captured poses don't fit the detected corners well -- "
          f"double check that camera's intrinsics file and recapture noisy samples.")


if __name__ == "__main__":
    main()