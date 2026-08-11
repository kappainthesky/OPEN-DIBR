# GStreamer Dual-Stream UDP Proof of Concept (OpenDIBR)

This folder contains a proof-of-concept for streaming RGB and depth video files over local UDP ports using GStreamer and receiving compressed HEVC buffers via GStreamer `appsink` in C++.

## Project Overview

- **`camera_config.json`**: Startup configuration file containing camera intrinsic/extrinsic parameters (following OpenDIBR format) and streaming settings (IP host, local UDP ports for color and depth streams).
- **`stream_sender.py`**: Python script using PyGObject GStreamer bindings to stream `v00_texture.mp4` (RGB) and `v00_depth.mp4` (Depth) via RTP HEVC payload over UDP ports `5000` and `5001`.
- **`verify_streams.sh`**: Shell script to verify both RGB and depth streams using `gst-launch-1.0` command line pipelines.
- **`receiver.cpp`**: Minimal C++ GStreamer application utilizing `appsink` to extract compressed HEVC buffers from incoming UDP RTP streams ready for future decoder integration with OpenDIBR.
- **`CMakeLists.txt`**: Build configuration for compiling `receiver.cpp`.

---

## How to Run

### Step 1: Verify Streams with `gst-launch-1.0`
Run the verification script to start the sender and verify incoming packets via `gst-launch-1.0`:
```bash
./gstreamer/verify_streams.sh
```

### Step 2: Build the C++ GStreamer Receiver
Build the receiver application using CMake:
```bash
cd gstreamer
mkdir -p build && cd build
cmake ..
make
```

### Step 3: Run the Stream Sender and C++ Receiver
In one terminal, start the Python stream sender:
```bash
python3 gstreamer/stream_sender.py gstreamer/camera_config.json
```

In another terminal, run the compiled C++ receiver:
```bash
./gstreamer/build/gstreamer_receiver gstreamer/camera_config.json
```

The C++ receiver will load `camera_config.json`, output camera parameters, and display incoming compressed HEVC buffers (`GstBuffer`) received via `appsink` for both RGB and Depth channels.
