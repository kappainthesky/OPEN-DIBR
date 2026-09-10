"""
Helper (D435) RGB intrinsic calibration using the printed ChArUco board.

Run this BEFORE helper_to_screen_calib.py and main_to_helper_calib.py.
Factory/SDK-reported intrinsics for the RGB sensor are OEM defaults, not
something Intel's on-chip/dynamic calibration tools touch (those target the
depth/stereo IR module). For anything needing real metric accuracy,
calibrate the RGB sensor yourself.

Uses the SAME print_board_main_helper.png / board_specs.json as
main_to_helper_calib.py -- no second calibration pattern needed. Just move
the (already-printed, flat, rigid) board around in front of this camera.

Procedure:
    1. Point the D435 at the board.
    2. Move the board (or the camera) to get diverse views: different
       distances, different tilt angles, and positions covering every
       corner of the frame, not just the center. 20-30 good captures is
       a solid target; more is better up to a point of diminishing returns.
    3. Press 'c' to capture a view once the board is detected with enough
       corners. Press 'q'/ESC once you have enough views to run calibration.

Output:
    helper_intrinsics.npz containing 'camera_matrix', 'dist_coeffs',
    'rms_error' (overall RMS reprojection error, scalar) and
    'per_view_errors' (per-view reprojection error, one value per
    captured view) -- the exact format both helper_to_screen_calib.py
    and main_to_helper_calib.py expect (they only read camera_matrix /
    dist_coeffs; the error fields are extra and safe to ignore there).

A per-view reprojection error and the overall RMS reprojection error are
printed at the end. Aim for well under 0.5 px RMS; if you're seeing more,
recapture with more diverse views (especially frame corners) or check that
the board is flat and the printed square size is accurate.
"""

import cv2
import numpy as np
import pyrealsense2 as rs

# ============================== CONFIG ======================================

# Must match print_board_main_helper.png / board_specs.json exactly
PRINT_DICT      = cv2.aruco.DICT_5X5_1000
PRINT_SQUARES_X = 8
PRINT_SQUARES_Y = 6
SQUARE_LEN_M    = 0.030   # verify against your printed board with a ruler
MARKER_LEN_M    = 0.022

DEVICE_NAME_SUBSTRING = "D435"
STREAM_W, STREAM_H, FPS = 640, 480, 30    # matched to MAIN_RES/MAIN_FPS so both cameras
                                           # calibrate/capture at the same resolution --
                                           # must also match HELPER_RES in main_to_helper_calib.py
                                           # and STREAM_W/H in helper_to_screen_calib.py if you
                                           # change it there too

OUTPUT_FILE = "helper_calib_intrinsics.npz"

MIN_CORNERS_TO_ACCEPT = 12       # higher bar than pose-estimation scripts -- intrinsic
                                  # calibration benefits from well-covered, confident views
N_VIEWS_TARGET        = 30       # auto-stop once this many captures are collected
MIN_VIEWS_TO_CALIBRATE = 8       # calibrateCamera will run below N_VIEWS_TARGET if you
                                  # quit early, but refuses below this floor

WARMUP_FRAMES = 30  # discard frames right after start() while auto-exposure/WB settle

# =============================================================================


def find_device_serial(name_substring):
    ctx = rs.context()
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        if name_substring.lower() in name.lower():
            return dev.get_info(rs.camera_info.serial_number), name
    return None, None


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(PRINT_DICT)
    board = cv2.aruco.CharucoBoard((PRINT_SQUARES_X, PRINT_SQUARES_Y),
                                    SQUARE_LEN_M, MARKER_LEN_M, aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    serial, name = find_device_serial(DEVICE_NAME_SUBSTRING)
    if serial is None:
        print(f"ERROR: no {DEVICE_NAME_SUBSTRING} found. Check it's connected.")
        return
    print(f"Camera: {name}  (serial {serial})")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
    pipeline.start(config)

    print(f"Warming up auto-exposure/white-balance ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()

    all_obj_points = []
    all_img_points = []
    image_size = None

    print("\nLive view starting. Move the board (or camera) around for diverse views:")
    print("different distances, tilt angles, and positions -- including frame corners.")
    print(f"Press 'c' to capture once the board is detected (need >= {MIN_CORNERS_TO_ACCEPT} corners).")
    print(f"Press 'q' or ESC to stop early and calibrate. Auto-stops at {N_VIEWS_TARGET} views.\n")

    try:
        while len(all_obj_points) < N_VIEWS_TARGET:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])  # (width, height)

            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

            # Normalize shapes -- OpenCV 5.0's detectBoard can return squeezed/
            # inconsistent shapes that make .total() disagree between corners
            # and ids even when len() matches, which trips an assertion in
            # drawDetectedCornersCharuco. This is a pure container reshape,
            # not a data change.
            if charuco_corners is not None:
                charuco_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
            if charuco_ids is not None:
                charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)

            vis = img.copy()
            corners_found = (charuco_ids is not None and charuco_corners is not None
                            and len(charuco_corners) == len(charuco_ids)
                            and len(charuco_ids) >= MIN_CORNERS_TO_ACCEPT
                            )

            if corners_found:
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                n = len(charuco_ids)
                cv2.putText(vis, f"corners: {n}  [c]=capture  views: {len(all_obj_points)}/{N_VIEWS_TARGET}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                n_detected = 0 if charuco_ids is None else len(charuco_ids)
                cv2.putText(vis, f"board not detected well enough ({n_detected} corners)  views: {len(all_obj_points)}/{N_VIEWS_TARGET}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("helper_intrinsic_calib", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and corners_found:
                obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                # Normalize shapes/dtype -- matchImagePoints can return squeezed
                # (N, 2)/(N, 3) arrays instead of (N, 1, 2)/(N, 1, 3), which later
                # causes a type mismatch (CV_32FC1 vs CV_32FC2) in cv2.norm during
                # per-view error computation. Pure container reshape, no data change.
                obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 1, 3)
                img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)
                all_obj_points.append(obj_points)
                all_img_points.append(img_points)
                print(f"  captured view {len(all_obj_points)}/{N_VIEWS_TARGET}  ({len(charuco_ids)} corners)")
            elif key in (ord('q'), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(all_obj_points) < MIN_VIEWS_TO_CALIBRATE:
        print(f"\nOnly {len(all_obj_points)} views captured (need >= {MIN_VIEWS_TO_CALIBRATE}). "
              f"Nothing saved -- rerun and capture more diverse views.")
        return

    print(f"\nRunning calibration on {len(all_obj_points)} views...")
    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, image_size, None, None)

    # Per-view reprojection error, so you can spot and discard any bad captures.
    # Uses the standard RMS-per-view formula (L2 norm over the sqrt of the point
    # count), matching how cv2.calibrateCamera computes the overall rms_error.
    per_view_errors = []
    for i in range(len(all_obj_points)):
        proj, _ = cv2.projectPoints(all_obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        img_pts = np.asarray(all_img_points[i], dtype=np.float32).reshape(-1, 1, 2)
        err = cv2.norm(img_pts, proj.reshape(-1, 1, 2), cv2.NORM_L2) / np.sqrt(len(proj))
        per_view_errors.append(err)
    per_view_errors = np.array(per_view_errors)

    np.savez(OUTPUT_FILE,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rms_error=rms_error,
             per_view_errors=per_view_errors)

    print(f"\nSaved {OUTPUT_FILE}")
    print(f"n_views = {len(all_obj_points)}")
    print(f"overall RMS reprojection error (px): {rms_error:.4f}")
    print(f"per-view reprojection error (px): {[f'{e:.3f}' for e in per_view_errors]}")
    print(f"\ncamera_matrix:\n{camera_matrix}")
    print(f"dist_coeffs: {dist_coeffs.flatten()}")
    print(f"\nAim for well under 0.5 px RMS. If it's higher, recapture with more "
          f"diverse views (especially frame corners/edges) or double check the "
          f"printed square size with a ruler.")


if __name__ == "__main__":
    main()