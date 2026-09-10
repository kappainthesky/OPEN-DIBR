#!/bin/bash
# ==============================================================================
# OpenDIBR: Live Intel RealSense RGB-D Rendering + Real-Time Eye Tracking
# (Combined 1-Camera RealSense D455 Live Input + eye3drgbd_updated Screen Calib)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"8

echo "==========================================================="
echo " Starting OpenDIBR Live RealSense + eye3drgbd_updated System"
echo " (1 RealSense D455 Camera + Screen-Calibrated Eye Tracking)"
echo "==========================================================="

# Select Python environment with mediapipe & pyrealsense2
if [ -f "$SCRIPT_DIR/eye_tracker/.venv/bin/python3" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/eye_tracker/.venv/bin/python3"
elif [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python3"
else
    PYTHON_EXEC="python3"
fi

cleanup() {
    echo ""
    echo "[Launcher] Stopping Eye Tracker..."88
    if [ -n "$EYE_PID" ]; then
        kill -SIGINT "$EYE_PID" 2>/dev/null || kill -9 "$EYE_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Launch eye_3d_rgbd_updated in background (connects to OpenDIBR RGB-D bridge on 127.0.0.1:9998)
echo "[1/2] Launching Screen-Calibrated Eye Tracker ($PYTHON_EXEC)..."
"$PYTHON_EXEC" "$SCRIPT_DIR/eye_tracker/eye_3d_screen.py" --bridge &
EYE_PID=$!
sleep 0.5

# 2. Launch OpenDIBR Live RealSense DIBR Renderer (1 Camera)
echo "[2/2] Launching OpenDIBR Live RealSense DIBR Renderer..."
./build/RealtimeDIBR --realsense -j examples/realsense/camera_config.json --auto_triangle_margin "$@"
8