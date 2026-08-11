import os
import sys
import json
import argparse
import math
import numpy as np
import cv2 as cv

def inspect_video(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    ret, sample_frame = cap.read()
    cap.release()

    if not ret or sample_frame is None:
        raise RuntimeError(f"Failed to read sample frame from: {video_path}")

    return {
        'path': video_path,
        'fps': fps if fps > 0 else 30.0,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'sample_frame': sample_frame
    }

def analyze_depth_type(sample_frame):
    if len(sample_frame.shape) == 2:
        return "Grayscale 8-bit", 8, 120.0

    b, g, r = sample_frame[:, :, 0], sample_frame[:, :, 1], sample_frame[:, :, 2]
    diff_bg = np.abs(b.astype(int) - g.astype(int))
    diff_gr = np.abs(g.astype(int) - r.astype(int))

    is_grayscale = (np.mean(diff_bg) < 2.0) and (np.mean(diff_gr) < 2.0)

    if is_grayscale:
        # Check depth noise variance
        grad_x = np.abs(sample_frame[1:, :, 0].astype(float) - sample_frame[:-1, :, 0].astype(float))
        avg_grad = np.mean(grad_x)
        if avg_grad < 2.0:
            return "Clean Synthetic Grayscale", 8, 15.0
        else:
            return "RealSense Grayscale Depth", 8, 120.0
    else:
        return "Colorized Depth Map", 8, 250.0

def process_and_align_videos(rgb_info, depth_info, target_fps, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    out_rgb_path = os.path.join(output_dir, "v00_texture.mp4")
    out_depth_path = os.path.join(output_dir, "v00_depth.mp4")

    # Determine target duration & frame count
    duration_rgb = rgb_info['frame_count'] / rgb_info['fps']
    duration_depth = depth_info['frame_count'] / depth_info['fps']
    max_duration = max(duration_rgb, duration_depth)
    target_frame_count = int(math.ceil(max_duration * target_fps))

    print(f" -> Target Resolution: {rgb_info['width']}x{rgb_info['height']}")
    print(f" -> Target FPS       : {target_fps:.1f}")
    print(f" -> Target Duration  : {max_duration:.2f} seconds ({target_frame_count} frames)")

    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    # Process RGB
    print(f" -> Processing RGB stream -> {out_rgb_path}")
    cap_rgb = cv.VideoCapture(rgb_info['path'])
    writer_rgb = cv.VideoWriter(out_rgb_path, fourcc, target_fps, (rgb_info['width'], rgb_info['height']))

    frames_rgb = []
    while True:
        ret, frame = cap_rgb.read()
        if not ret:
            break
        if (frame.shape[1], frame.shape[0]) != (rgb_info['width'], rgb_info['height']):
            frame = cv.resize(frame, (rgb_info['width'], rgb_info['height']))
        frames_rgb.append(frame)
    cap_rgb.release()

    if not frames_rgb:
        raise RuntimeError("RGB video contained no valid frames.")

    for i in range(target_frame_count):
        frame = frames_rgb[i % len(frames_rgb)]
        writer_rgb.write(frame)
    writer_rgb.release()

    # Process Depth
    print(f" -> Processing Depth stream -> {out_depth_path}")
    cap_depth = cv.VideoCapture(depth_info['path'])
    writer_depth = cv.VideoWriter(out_depth_path, fourcc, target_fps, (rgb_info['width'], rgb_info['height']))

    frames_depth = []
    while True:
        ret, frame = cap_depth.read()
        if not ret:
            break
        if (frame.shape[1], frame.shape[0]) != (rgb_info['width'], rgb_info['height']):
            frame = cv.resize(frame, (rgb_info['width'], rgb_info['height']))
        frames_depth.append(frame)
    cap_depth.release()

    if not frames_depth:
        raise RuntimeError("Depth video contained no valid frames.")

    for i in range(target_frame_count):
        frame = frames_depth[i % len(frames_depth)]
        writer_depth.write(frame)
    writer_depth.release()

    return out_rgb_path, out_depth_path, target_frame_count

def generate_opendibr_json(output_dir, width, height, near, far, fov_deg, pos, rot, bitdepth_depth):
    # Focal length calculation from FOV
    fov_rad = math.radians(fov_deg)
    focal_x = (width / 2.0) / math.tan(fov_rad / 2.0)
    focal_y = focal_x
    cx = width / 2.0
    cy = height / 2.0

    json_path = os.path.join(output_dir, "dataset_config.json")

    config_data = {
        "Axial_system": "OPENGL",
        "cameras": [
            {
                "NameColor": "viewport",
                "Position": [pos[0], pos[1], pos[2]],
                "Rotation": [rot[0], rot[1], rot[2]],
                "Depth_range": [near, far],
                "Resolution": [width, height],
                "Projection": "Perspective",
                "Focal": [round(focal_x, 2), round(focal_y, 2)],
                "Principle_point": [round(cx, 1), round(cy, 1)]
            },
            {
                "NameColor": "v00_texture.mp4",
                "NameDepth": "v00_depth.mp4",
                "Position": [pos[0], pos[1], pos[2]],
                "Rotation": [rot[0], rot[1], rot[2]],
                "Depth_range": [near, far],
                "Resolution": [width, height],
                "Projection": "Perspective",
                "Focal": [round(focal_x, 2), round(focal_y, 2)],
                "Principle_point": [round(cx, 1), round(cy, 1)],
                "BitDepthColor": 8,
                "BitDepthDepth": bitdepth_depth
            }
        ]
    }

    with open(json_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    return json_path

def main():
    parser = argparse.ArgumentParser(description="Automatic Open-DIBR Dataset Preparation Tool")
    parser.add_argument("--rgb_video", required=True, help="Path to input RGB video file")
    parser.add_argument("--depth_video", required=True, help="Path to input Depth video file")
    parser.add_argument("--output_dir", required=True, help="Output dataset directory path")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera Horizontal FOV in degrees (default: 60.0)")
    parser.add_argument("--near", type=float, default=0.3, help="Near depth clipping plane in meters (default: 0.3)")
    parser.add_argument("--far", type=float, default=10.0, help="Far depth clipping plane in meters (default: 10.0)")
    parser.add_argument("--pos", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Camera 3D position X Y Z")
    parser.add_argument("--rot", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Camera 3D rotation Pitch Yaw Roll")
    parser.add_argument("--target_fps", type=float, default=30.0, help="Target uniform FPS (default: 30.0)")

    args = parser.parse_args()

    print("=============================================")
    print(" [Open-DIBR] Automated Dataset Preparation")
    print("=============================================")

    # Step 1: Validate Videos
    print("\n[Step 1/5] Validating Input Streams...")
    rgb_info = inspect_video(args.rgb_video)
    depth_info = inspect_video(args.depth_video)

    print(f" -> RGB Video  : {rgb_info['width']}x{rgb_info['height']} @ {rgb_info['fps']:.1f} FPS ({rgb_info['frame_count']} frames)")
    print(f" -> Depth Video: {depth_info['width']}x{depth_info['height']} @ {depth_info['fps']:.1f} FPS ({depth_info['frame_count']} frames)")

    # Step 2: Depth Format & Margin Detection
    print("\n[Step 2/5] Inspecting Depth Characteristics...")
    depth_type, bitdepth, recommended_margin = analyze_depth_type(depth_info['sample_frame'])
    print(f" -> Detected Depth Format        : {depth_type}")
    print(f" -> Recommended Deletion Margin : {recommended_margin:.1f}")

    # Step 3: Align & Extend Videos
    print("\n[Step 3/5] Standardizing FPS & Extending Video Streams...")
    out_rgb, out_depth, final_frames = process_and_align_videos(rgb_info, depth_info, args.target_fps, args.output_dir)

    # Step 4: Generate Open-DIBR JSON
    print("\n[Step 4/5] Generating Open-DIBR Camera Parameters JSON...")
    json_path = generate_opendibr_json(
        args.output_dir, rgb_info['width'], rgb_info['height'],
        args.near, args.far, args.fov, args.pos, args.rot, bitdepth
    )
    print(f" -> Created JSON Config: {json_path}")

    # Step 5: Summary
    print("\n=============================================")
    print(" [SUCCESS] Open-DIBR Dataset Preparation Complete!")
    print("=============================================")
    print(f" Dataset Folder        : {args.output_dir}")
    print(f" Color Stream          : v00_texture.mp4")
    print(f" Depth Stream          : v00_depth.mp4")
    print(f" JSON Config           : dataset_config.json")
    print(f" Total Frames          : {final_frames}")
    print(f" Recommended Margin    : {recommended_margin:.1f}")
    print("\nTo render this dataset in Open-DIBR, run:")
    print(f"  ./build/RealtimeDIBR -i {args.output_dir} -j {json_path} --triangle_deletion_margin {recommended_margin:.1f} --auto_triangle_margin\n")

if __name__ == "__main__":
    main()
