

import cv2
import numpy as np
import json
import os
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as Rot

# ============================== CONFIG ======================================

# Must match print_board_main_helper.png / board_specs.json exactly
PRINT_DICT      = cv2.aruco.DICT_5X5_1000
PRINT_SQUARES_X = 8
PRINT_SQUARES_Y = 6
SQUARE_LEN_M    = 0.030   # verify against your printed board with a ruler
MARKER_LEN_M    = 0.022

MAIN_SERIAL      = "215122252978"
SECONDARY_SERIAL = "215122255078"

MAIN_INTRINSICS_FILE      = "main_calib_intrinsics_factory.npz"
SECONDARY_INTRINSICS_FILE = "secondary_intrinsics_factory.npz"

MAIN_RES        = (640, 480)   # must match main_calib_intrinsics.npz
MAIN_FPS        = 30
SECONDARY_RES   = (640, 480)   # must match secondary_calib_intrinsics.npz
SECONDARY_FPS   = 30

OUTPUT_FILE = "main_to_secondary_calib.json"

MIN_CORNERS_TO_ACCEPT = 8
N_SAMPLES_TARGET = 15

WARMUP_FRAMES = 30  # discard this many frames after pipeline.start() for auto-exposure/WB to settle

# --- Rotation output convention -- UNVERIFIED, see warning in docstring ---
# scipy's Rotation.as_euler() intrinsic/extrinsic sequence string, e.g.
# "xyz", "XYZ", "zyx", "ZYX". Case matters (lower = extrinsic, upper =
# intrinsic). This is a GUESS -- confirm against camera_config.json's
# loader before trusting the printed/saved Rotation vector.
ROTATION_EULER_SEQ     = "xyz"
ROTATION_OUTPUT_DEGREES = True   # True -> degrees, False -> radians



def find_device_by_serial(serial):
    """Confirms the target serial is actually connected and returns its name."""
    ctx = rs.context()
    for dev in ctx.query_devices():
        dev_serial = dev.get_info(rs.camera_info.serial_number)
        if dev_serial == serial:
            return dev_serial, dev.get_info(rs.camera_info.name)
    return None, None


def load_intrinsics(path):
    data = np.load(path)
    return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)


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


def rotation_matrix_to_config_vector(R_ms):
    """
    Convert a 3x3 rotation matrix to the 3-value vector camera_config.json
    expects, using ROTATION_EULER_SEQ / ROTATION_OUTPUT_DEGREES above.

    UNVERIFIED CONVENTION -- see the docstring warning at the top of this
    file. Confirm ROTATION_EULER_SEQ against the config-loading code (or
    ask Rizal) before trusting these numbers in production.
    """
    euler = Rot.from_matrix(R_ms).as_euler(ROTATION_EULER_SEQ, degrees=ROTATION_OUTPUT_DEGREES)
    return euler.tolist()


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(PRINT_DICT)
    board = cv2.aruco.CharucoBoard((PRINT_SQUARES_X, PRINT_SQUARES_Y),
                                    SQUARE_LEN_M, MARKER_LEN_M, aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    main_serial, main_name = find_device_by_serial(MAIN_SERIAL)
    secondary_serial, secondary_name = find_device_by_serial(SECONDARY_SERIAL)

    if main_serial is None:
        print(f"ERROR: no device with serial {MAIN_SERIAL} found. Check it's connected.")
        return
    if secondary_serial is None:
        print(f"ERROR: no device with serial {SECONDARY_SERIAL} found. Check it's connected.")
        return

    print(f"Main camera:      {main_name}  (serial {main_serial})")
    print(f"Secondary camera: {secondary_name}  (serial {secondary_serial})")

    K_main, dist_main = load_intrinsics(MAIN_INTRINSICS_FILE)
    K_secondary, dist_secondary = load_intrinsics(SECONDARY_INTRINSICS_FILE)

    input("\nPlace the board in its fixed position, then press Enter to start with the MAIN camera...")

    R_main, t_main, n_main, std_main, reproj_main = capture_pose(
        main_serial, "MAIN", MAIN_RES[0], MAIN_RES[1], MAIN_FPS, K_main, dist_main, board, detector)

    if R_main is None:
        print("No samples captured from main camera. Aborting.")
        return

    print(f"\nMain camera done: {n_main} samples, translation std (mm): {std_main}, "
          f"mean reproj error (px): {reproj_main['mean_reproj_error_px']:.4f}, "
          f"max reproj error (px): {reproj_main['max_reproj_error_px']:.4f}")
    input("\nDO NOT MOVE THE BOARD. Press Enter to continue with the SECONDARY camera...")

    R_secondary, t_secondary, n_secondary, std_secondary, reproj_secondary = capture_pose(
        secondary_serial, "SECONDARY", SECONDARY_RES[0], SECONDARY_RES[1], SECONDARY_FPS,
        K_secondary, dist_secondary, board, detector)

    if R_secondary is None:
        print("No samples captured from secondary camera. Aborting.")
        return

    print(f"\nSecondary camera done: {n_secondary} samples, translation std (mm): {std_secondary}, "
          f"mean reproj error (px): {reproj_secondary['mean_reproj_error_px']:.4f}, "
          f"max reproj error (px): {reproj_secondary['max_reproj_error_px']:.4f}")

    # T_main->secondary = T_main->board * inv(T_secondary->board)
    R_ms = R_main @ R_secondary.T
    t_ms = t_main - R_ms @ t_secondary

    # camera_config.json-formatted block
    position_vec = t_ms.tolist()
    rotation_vec = rotation_matrix_to_config_vector(R_ms)

    camera_config_block = {
        "Position": position_vec,
        "Rotation": rotation_vec,
    }

    result = {
        "R": R_ms.tolist(),
        "t": t_ms.tolist(),
        "n_samples_main": n_main,
        "n_samples_secondary": n_secondary,
        "translation_std_mm_main": std_main,
        "translation_std_mm_secondary": std_secondary,
        "reproj_error_main": reproj_main,
        "reproj_error_secondary": reproj_secondary,
        "camera_config_block": camera_config_block,
        "rotation_convention_used": {
            "euler_seq": ROTATION_EULER_SEQ,
            "degrees": ROTATION_OUTPUT_DEGREES,
            "verified": False,
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {OUTPUT_FILE}")
    print(f"T_main->secondary rotation (raw 3x3 matrix, for sanity-checking the Euler conversion below):\n{R_ms}")
    print(f"T_main->secondary translation (m): {t_ms}")
    print(f"\nIf either translation_std is more than a few mm, recapture that "
          f"camera's samples -- likely an unstable hand-hold or the board "
          f"partially leaving frame during capture.")
    print(f"If either mean/max reprojection error is high (aim for well under 1 px), "
          f"that camera's captured poses don't fit the detected corners well -- "
          f"double check that camera's intrinsics file and recapture noisy samples.")

    print("\n" + "=" * 60)
    print("camera_config.json block (paste in for the secondary camera):")
    print("=" * 60)
    print(json.dumps(camera_config_block, indent=6))
    print("=" * 60)
    print(f"ROTATION CONVENTION USED: euler_seq='{ROTATION_EULER_SEQ}', "
          f"degrees={ROTATION_OUTPUT_DEGREES} -- THIS IS UNVERIFIED.")
    print("Before trusting this Rotation vector, confirm it against the actual "
          "config-loading code (wherever 'Position'/'Rotation' are parsed "
          "alongside 'Focal'), or ask Rizal. If it's wrong, recompute manually "
          "from the raw R matrix printed above using the correct convention -- "
          "do not just try sequences until numbers 'look plausible'.")


if __name__ == "__main__":
    main()