"""
Save factory RGB intrinsics for a specific RealSense device, identified by
SERIAL NUMBER rather than device name -- for setups like OpenDIBR's config
where a camera is referenced by role/serial rather than model name, e.g.:

    "NameColor": "realsense_secondary",
    "NameDepth": "realsense_secondary",
    "SerialNumber": "215122255078",
    "Role": "render_secondary"

Saves to the SAME .npz format as the other calibration scripts
(camera_matrix, dist_coeffs, ...) so it's a drop-in wherever those are read.

Usage:
    python secondary_rgbd_intrinsic.py
    python secondary_rgbd_intrinsic.py --serial 215122255078 --role secondary
    python secondary_rgbd_intrinsic.py --serial 215122255078 --out secondary_calib_intrinsics.npz
"""

import argparse
import numpy as np
import pyrealsense2 as rs

DEFAULT_SERIAL = "215122255078"
DEFAULT_ROLE = "secondary"


def find_device_by_serial(serial):
    ctx = rs.context()
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return dev
    return None


def main():
    parser = argparse.ArgumentParser(description="Save factory RGB intrinsics for a device identified by serial number")
    parser.add_argument("--serial", type=str, default=DEFAULT_SERIAL,
                         help=f"Device serial number (default: {DEFAULT_SERIAL})")
    parser.add_argument("--role", type=str, default=DEFAULT_ROLE,
                         help="Role/name label, used only to build the default output filename")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, default=None,
                         help="Output .npz path (default: '<role>_intrinsics_factory.npz')")
    args = parser.parse_args()

    out_path = args.out or f"{args.role}_intrinsics_factory.npz"

    dev = find_device_by_serial(args.serial)
    if dev is None:
        available = [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
                     for d in rs.context().query_devices()]
        print(f"ERROR: no device with serial '{args.serial}' found.")
        if available:
            print("Connected device(s):")
            for name, serial in available:
                print(f"  {name}  (serial {serial})")
        else:
            print("No RealSense devices detected at all -- check connections.")
        return

    name = dev.get_info(rs.camera_info.name)
    print(f"Camera: {name}  (serial {args.serial}, role: {args.role})")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
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

    np.savez(out_path,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rms_error=np.nan,
             per_view_errors=np.array([]),
             source="factory",
             serial=args.serial,
             role=args.role,
             width=args.width, height=args.height, fps=args.fps)

    print(f"\nSaved {out_path}")
    print(f"camera_matrix:\n{camera_matrix}")
    print(f"dist_coeffs: {dist_coeffs.flatten()}")
    print(f"\nFixed for this exact device (S/N {args.serial}) at {args.width}x{args.height}@{args.fps}fps.")


if __name__ == "__main__":
    main()