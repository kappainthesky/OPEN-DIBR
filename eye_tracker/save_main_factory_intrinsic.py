"""
Save the RealSense D455's factory RGB intrinsics to an .npz file in the
SAME format main_calib_intrinsics.py produces, so it's a drop-in replacement
for main_calib_intrinsics.npz in main_to_helper_calib.py / eye_tracker.py.

Factory intrinsics are fixed per physical unit + resolution/FPS (baked into
the camera's firmware), so this just needs to be run once per device/config
-- it won't change between runs on the same D455 unit at the same settings.

Usage:
    python save_factory_intrinsics.py
    python save_factory_intrinsics.py --width 640 --height 480 --fps 30 --out main_calib_intrinsics.npz
"""

import argparse
import numpy as np
import pyrealsense2 as rs

DEVICE_NAME_SUBSTRING = "D455"


def find_device_serial(name_substring):
    ctx = rs.context()
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        if name_substring.lower() in name.lower():
            return dev.get_info(rs.camera_info.serial_number), name
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Save D455 factory RGB intrinsics to .npz")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, default="main_calib_intrinsics_factory.npz",
                         help="Output path (default: main_calib_intrinsics_factory.npz -- "
                              "rename/point your pipeline at this, or pass "
                              "main_calib_intrinsics.npz to overwrite the calibrated one directly)")
    args = parser.parse_args()

    serial, name = find_device_serial(DEVICE_NAME_SUBSTRING)
    if serial is None:
        print(f"ERROR: no {DEVICE_NAME_SUBSTRING} found. Check it's connected.")
        return
    print(f"Camera: {name}  (serial {serial})")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    try:
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    finally:
        pipeline.stop()

    camera_matrix = np.array([
        [intr.fx, 0.0,     intr.ppx],
        [0.0,     intr.fy, intr.ppy],
        [0.0,     0.0,     1.0],
    ], dtype=np.float64)
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64).reshape(1, -1)

    # rms_error / per_view_errors don't apply to factory intrinsics (no
    # calibration was run) -- store None/empty so downstream code that
    # checks for these keys (main_to_helper_calib.py only reads
    # camera_matrix/dist_coeffs anyway) doesn't choke on a missing key.
    np.savez(args.out,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rms_error=np.nan,
             per_view_errors=np.array([]),
             source="factory",
             serial=serial,
             width=args.width, height=args.height, fps=args.fps)

    print(f"\nSaved {args.out}")
    print(f"camera_matrix:\n{camera_matrix}")
    print(f"dist_coeffs: {dist_coeffs.flatten()}")
    print(f"\nThese are fixed for this exact device (S/N {serial}) at "
          f"{args.width}x{args.height}@{args.fps}fps -- re-run this script only "
          f"if you change resolution/FPS or swap to a different physical unit.")


if __name__ == "__main__":
    main()