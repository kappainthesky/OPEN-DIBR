"""
Full-rig detection diagnostic -- runs ALL THREE checks simultaneously, each
with full stage1 (raw marker)/stage2 (charuco corner) diagnostic detail:

    1. MAIN   (D455) detecting the PRINTED board
    2. HELPER (D435) detecting the PRINTED board
    3. HELPER (D435) detecting the SCREEN board

Two windows: MAIN (D455) and HELPER (D435). The helper window checks BOTH
board types on the same live feed and shows diagnostics for each.

For each check, shows:
    - stage1: raw ArUco marker count (0 = nothing decoded at all --
      too far/blurry/glare/wrong dictionary/out of frame)
    - stage2: charuco interpolated corner count (markers found but 0 here
      = too scattered/occluded for corner interpolation)
    - brightness / blur (Laplacian variance) image-quality metrics
    - a one-line diagnosis

Run this once your physical setup (board placement, screen, both camera
mounts) is roughly where you want it, to confirm all three actually DETECT
(not just visually see) their target before running the real calibration
scripts.

Controls:
    q / ESC (either window) -> quit both
"""

import cv2
import numpy as np
import pyrealsense2 as rs

# CONFIG

BOARDS = {
    "PRINT": dict(
        dict_id=cv2.aruco.DICT_5X5_1000,
        squares_x=8, squares_y=6,
        square_len_m=0.030, marker_len_m=0.022,
    ),
    "SCREEN": dict(
        dict_id=cv2.aruco.DICT_4X4_50,
        squares_x=10, squares_y=6,
        square_len_m=0.025, marker_len_m=0.01875,
    ),
}

MAIN_RES,   MAIN_FPS   = (640, 480),  30
HELPER_RES, HELPER_FPS = (640, 480),  30   # matched to MAIN_RES/MAIN_FPS -- keep in sync with
                                            # STREAM_W/H in helper_intrinsic_calib.py,
                                            # HELPER_RES in main_to_helper_calib.py, and
                                            # STREAM_W/H in helper_to_screen_calib.py

MIN_CORNERS_TO_ACCEPT = 8



def find_device_serial(name_substring):
    ctx = rs.context()
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        if name_substring.lower() in name.lower():
            return dev.get_info(rs.camera_info.serial_number), name
    return None, None


def build_detector(cfg):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cfg["dict_id"])
    board = cv2.aruco.CharucoBoard(
        (cfg["squares_x"], cfg["squares_y"]), cfg["square_len_m"], cfg["marker_len_m"], aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    marker_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    return charuco_detector, marker_detector, board


def start_pipeline_safe(pipeline, config, serial, label):
    """Start a pipeline with clearer diagnostics than the bare RealSense error.

    'Couldn't resolve requests' from pipeline.start() almost always means either:
      (a) the device is still held by another process (e.g. a previous script/app
          that didn't cleanly release it), or
      (b) the requested width/height/fps/format combo isn't supported.
    This tries once, and on failure attempts a hardware reset of just that
    device and retries once before giving up with an actionable message.
    """
    try:
        return pipeline.start(config)
    except RuntimeError as e:
        print(f"\n[{label}] pipeline.start() failed: {e}")
        print(f"[{label}] Most likely cause: the device (serial {serial}) is still held by "
              f"another process (e.g. a previous run that didn't release it cleanly), "
              f"or this resolution/fps/format isn't supported by this stream.")
        print(f"[{label}] Checking for other processes... run `lsof | grep video` or "
              f"`ps aux | grep python` in another terminal to look for a stale process "
              f"still holding the camera.")
        print(f"[{label}] Attempting a hardware reset of this device and one retry...")
        ctx = rs.context()
        for dev in ctx.query_devices():
            if dev.get_info(rs.camera_info.serial_number) == serial:
                dev.hardware_reset()
                break
        import time
        time.sleep(2.0)  # give the device time to re-enumerate on the bus after reset
        try:
            return pipeline.start(config)
        except RuntimeError as e2:
            print(f"[{label}] Retry also failed: {e2}")
            print(f"[{label}] Unplug/replug the device, confirm no other script/app has it "
                  f"open, and rerun.")
            raise


def image_quality_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return brightness, blur


def run_diagnostic(img, vis, charuco_detector, marker_detector, label, y_start):
    """Runs stage1+stage2 detection for one board type, draws overlays, returns
    (n_markers, n_charuco, diagnosis_line) and writes text starting at y_start."""
    marker_corners, marker_ids, rejected = marker_detector.detectMarkers(img)
    n_markers = 0 if marker_ids is None else len(marker_ids)
    if n_markers > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(img)

    # Normalize shapes -- OpenCV 5.0's detectBoard can return squeezed/
    # inconsistent shapes that make .total() disagree between corners and
    # ids even when len() matches, which trips an assertion in
    # drawDetectedCornersCharuco. Pure container reshape, no data change.
    if charuco_corners is not None:
        charuco_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    if charuco_ids is not None:
        charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)

    n_charuco = 0 if charuco_ids is None else len(charuco_ids)
    if n_charuco > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    if n_markers == 0:
        diag = "no markers -> too far/blurry/glare/wrong dict/out of frame"
        ok = False
    elif n_charuco == 0:
        diag = "markers found, 0 charuco corners -> scattered/occluded"
        ok = False
    elif n_charuco < MIN_CORNERS_TO_ACCEPT:
        diag = f"only {n_charuco} corners -> move closer / reduce angle"
        ok = False
    else:
        diag = "OK -- detects reliably"
        ok = True

    lines = [
        f"[{label}] markers:{n_markers}  charuco:{n_charuco}  -- {diag}",
    ]
    color = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(vis, lines[0], (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return n_markers, n_charuco, ok


def main():
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

    print_charuco, print_marker, print_board = build_detector(BOARDS["PRINT"])
    screen_charuco, screen_marker, screen_board = build_detector(BOARDS["SCREEN"])

    main_pipeline = rs.pipeline()
    main_config = rs.config()
    main_config.enable_device(main_serial)
    main_config.enable_stream(rs.stream.color, MAIN_RES[0], MAIN_RES[1], rs.format.bgr8, MAIN_FPS)
    start_pipeline_safe(main_pipeline, main_config, main_serial, "MAIN")

    helper_pipeline = rs.pipeline()
    helper_config = rs.config()
    helper_config.enable_device(helper_serial)
    helper_config.enable_stream(rs.stream.color, HELPER_RES[0], HELPER_RES[1], rs.format.bgr8, HELPER_FPS)
    start_pipeline_safe(helper_pipeline, helper_config, helper_serial, "HELPER")

    print("\nLive view starting -- two windows.")
    print("Checking simultaneously:")
    print("  MAIN window:   PRINT board detection (stage1 markers / stage2 charuco corners)")
    print("  HELPER window: PRINT board AND SCREEN board detection, same feed")
    print("\nAll three lines should read 'OK -- detects reliably' before you trust the real")
    print("calibration scripts to work from this physical setup.")
    print("Press 'q' or ESC in either window to quit.\n")

    try:
        while True:
            main_frames = main_pipeline.wait_for_frames()
            main_color = main_frames.get_color_frame()
            helper_frames = helper_pipeline.wait_for_frames()
            helper_color = helper_frames.get_color_frame()

            if main_color:
                main_img = np.asanyarray(main_color.get_data())
                main_vis = main_img.copy()
                brightness, blur = image_quality_metrics(main_img)
                run_diagnostic(main_img, main_vis, print_charuco, print_marker, "MAIN sees PRINT", 30)
                cv2.putText(main_vis, f"brightness:{brightness:.0f} blur:{blur:.0f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("MAIN (D455)", main_vis)

            if helper_color:
                helper_img = np.asanyarray(helper_color.get_data())
                helper_vis = helper_img.copy()
                brightness, blur = image_quality_metrics(helper_img)
                run_diagnostic(helper_img, helper_vis, print_charuco, print_marker, "HELPER sees PRINT", 30)
                run_diagnostic(helper_img, helper_vis, screen_charuco, screen_marker, "HELPER sees SCREEN", 60)
                cv2.putText(helper_vis, f"brightness:{brightness:.0f} blur:{blur:.0f}",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("HELPER (D435)", helper_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        main_pipeline.stop()
        helper_pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()