#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COLOR_PORT=5000
DEPTH_PORT=5001

echo "=========================================================================="
echo "GStreamer Proof-of-Concept: Verifying Streams with gst-launch-1.0"
echo "=========================================================================="

echo "[1/4] Starting stream sender in background using camera_config.json..."
python3 "$SCRIPT_DIR/stream_sender.py" "$SCRIPT_DIR/camera_config.json" &
SENDER_PID=$!

cleanup() {
    echo "[Clean] Terminating sender background process (PID $SENDER_PID)..."
    kill $SENDER_PID 2>/dev/null || true
    wait $SENDER_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for sender pipelines to initialize..."
sleep 2

echo "--------------------------------------------------------------------------"
echo "[2/4] Verifying Color Stream on TCP Port $COLOR_PORT with gst-launch-1.0..."
echo "--------------------------------------------------------------------------"
gst-launch-1.0 -v \
  tcpclientsrc host=127.0.0.1 port=$COLOR_PORT timeout=5 ! \
  h265parse ! \
  fakesink dump=true num-buffers=5

echo "--------------------------------------------------------------------------"
echo "[3/4] Verifying Depth Stream on TCP Port $DEPTH_PORT with gst-launch-1.0..."
echo "--------------------------------------------------------------------------"
gst-launch-1.0 -v \
  tcpclientsrc host=127.0.0.1 port=$DEPTH_PORT timeout=5 ! \
  h265parse ! \
  fakesink dump=true num-buffers=5

echo "=========================================================================="
echo "[4/4] SUCCESS: Both RGB and Depth TCP streams verified with gst-launch-1.0!"
echo "=========================================================================="
