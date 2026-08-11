#!/usr/bin/env python3
"""
Multi-camera Synchronized Stream Sender for OpenDIBR.

Streams all input cameras defined in camera_config.json in parallel over UDP RTP:
- Camera 0: Color port 5000, Depth port 5001
- Camera 1: Color port 5002, Depth port 5003
- Camera 2: Color port 5004, Depth port 5005
- Camera 3: Color port 5006, Depth port 5007
"""

import sys
import os
import json
import time
import signal
import argparse
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

class SynchronizedStreamSender:
    def __init__(self, config_path, loop=True):
        Gst.init(None)

        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)

        print(f"[Sender] Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        streaming_cfg = self.config.get("streaming", {})
        self.host = streaming_cfg.get("host", "127.0.0.1")
        self.base_color_port = streaming_cfg.get("color_port", 5000)
        self.base_depth_port = streaming_cfg.get("depth_port", 5001)
        self.loop = loop

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Parse cameras (skip viewport camera 0)
        cameras = self.config.get("cameras", [])
        self.stream_targets = []
        
        cam_idx = 0
        for cam in cameras:
            name_color = cam.get("NameColor", "")
            name_depth = cam.get("NameDepth", "")
            if not name_depth: # skip viewport
                continue
                
            color_path = name_color if os.path.isabs(name_color) else os.path.join(repo_root, "examples/Fan", name_color)
            depth_path = name_depth if os.path.isabs(name_depth) else os.path.join(repo_root, "examples/Fan", name_depth)
            
            if not os.path.exists(color_path):
                # Try examples/Fan
                color_path = os.path.join(repo_root, "examples/Fan", os.path.basename(name_color))
            if not os.path.exists(depth_path):
                depth_path = os.path.join(repo_root, "examples/Fan", os.path.basename(name_depth))
                
            if not os.path.exists(color_path):
                raise FileNotFoundError(f"Color video file not found: {color_path}")
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth video file not found: {depth_path}")

            c_port = self.base_color_port + 2 * cam_idx
            d_port = self.base_depth_port + 2 * cam_idx
            
            self.stream_targets.append({
                "cam_idx": cam_idx,
                "color_file": color_path,
                "depth_file": depth_path,
                "color_port": c_port,
                "depth_port": d_port,
                "color_ssrc": 111111 + cam_idx * 2,
                "depth_ssrc": 222222 + cam_idx * 2,
                "last_color_seq": 1000 + cam_idx * 10000,
                "last_color_ts": 160000,
                "last_depth_seq": 2000 + cam_idx * 10000,
                "last_depth_ts": 160000,
                "color_seqnum_offset": 1000 + cam_idx * 10000,
                "color_timestamp_offset": 160000,
                "depth_seqnum_offset": 2000 + cam_idx * 10000,
                "depth_timestamp_offset": 160000,
            })
            cam_idx += 1

        self.loop_count = 0
        self.start_time = time.time()
        self.is_restarting = False

        print(f"[Sender] Target Host: {self.host}")
        print(f"[Sender] Total Active Streaming Cameras: {len(self.stream_targets)}")
        for target in self.stream_targets:
            print(f"  - Cam [{target['cam_idx']}]: Color port {target['color_port']} ({os.path.basename(target['color_file'])}), "
                  f"Depth port {target['depth_port']} ({os.path.basename(target['depth_file'])})")

        self.loop_engine = GLib.MainLoop()
        self.pipeline = None
        self.build_pipeline()

    def _make_probe_cb(self, target_dict, is_depth=False):
        def _probe_cb(pad, info):
            buf_list = info.get_buffer_list()
            if buf_list:
                n = buf_list.length()
                if n > 0:
                    last_buf = buf_list.get(n - 1)
                    res, map_info = last_buf.map(Gst.MapFlags.READ)
                    data = bytes(map_info.data)
                    if len(data) >= 12:
                        seq = (data[2] << 8) | data[3]
                        ts = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
                        if is_depth:
                            target_dict["last_depth_seq"] = seq
                            target_dict["last_depth_ts"] = ts
                        else:
                            target_dict["last_color_seq"] = seq
                            target_dict["last_color_ts"] = ts
                    last_buf.unmap(map_info)
            else:
                buf = info.get_buffer()
                if buf:
                    res, map_info = buf.map(Gst.MapFlags.READ)
                    data = bytes(map_info.data)
                    if len(data) >= 12:
                        seq = (data[2] << 8) | data[3]
                        ts = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
                        if is_depth:
                            target_dict["last_depth_seq"] = seq
                            target_dict["last_depth_ts"] = ts
                        else:
                            target_dict["last_color_seq"] = seq
                            target_dict["last_color_ts"] = ts
                    buf.unmap(map_info)
            return Gst.PadProbeReturn.OK
        return _probe_cb

    def build_pipeline(self):
        pipeline_elements = []
        for t in self.stream_targets:
            idx = t["cam_idx"]
            c_port = t["color_port"]
            d_port = t["depth_port"]

            color_sock = "/tmp/dibr_color.sock" if c_port == 5000 else f"/tmp/dibr_color_{c_port}.sock"
            depth_sock = "/tmp/dibr_depth.sock" if d_port == 5001 else f"/tmp/dibr_depth_{d_port}.sock"

            color_str = (
                f"filesrc location=\"{t['color_file']}\" ! qtdemux ! h265parse config-interval=1 ! "
                f"video/x-h265, stream-format=byte-stream, alignment=au ! "
                f"tcpserversink host=0.0.0.0 port={c_port} sync=true sync-method=latest-keyframe"
            )
            depth_str = (
                f"filesrc location=\"{t['depth_file']}\" ! qtdemux ! h265parse config-interval=1 ! "
                f"video/x-h265, stream-format=byte-stream, alignment=au ! "
                f"tcpserversink host=0.0.0.0 port={d_port} sync=true sync-method=latest-keyframe"
            )
            pipeline_elements.append(color_str)
            pipeline_elements.append(depth_str)

        combined_pipeline_str = " ".join(pipeline_elements)
        self.pipeline = Gst.parse_launch(combined_pipeline_str)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self.on_eos)
        bus.connect("message::error", self.on_error)

    def on_eos(self, bus, message):
        source_name = message.src.get_name() if message.src else "Pipeline"
        if not self.loop:
            print(f"[Sender] [{source_name}] Reached EOS in Single-Pass mode. Stopping...")
            self.stop()
            return

        self.loop_count += 1
        eos_time = time.time() - self.start_time
        print(f"[Sender] [{source_name}] Reached EOS at T={eos_time:.2f}s (Loop #{self.loop_count}). Seeking pipeline back to start (0)...")
        self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)

    def on_error(self, bus, message):
        err, debug = message.parse_error()
        source_name = message.src.get_name() if message.src else "Pipeline"
        print(f"[Sender] [{source_name}] Error: {err.message}", file=sys.stderr)
        if debug:
            print(f"[Sender] [{source_name}] Debug: {debug}", file=sys.stderr)
        self.stop()

    def start(self, startup_delay=0.3):
        print(f"[Sender] Starting Multi-Camera Synchronized GStreamer UDP streams at 30 fps...")
        if startup_delay > 0:
            print(f"[Sender] Waiting {startup_delay:.2f}s startup delay for receivers to bind sockets...")
            time.sleep(startup_delay)
        print("[Sender] Press Ctrl+C to stop.")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop_engine.run()
        except KeyboardInterrupt:
            print("\n[Sender] Interrupted by user (Ctrl+C).")
        finally:
            self.stop()

    def stop(self):
        print("[Sender] Stopping GStreamer pipeline...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop_engine and self.loop_engine.is_running():
            self.loop_engine.quit()

def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Continuous Monotonic RTP Stream Sender")
    parser.add_argument("config", nargs="?", default="gstreamer/camera_config.json", help="Path to config JSON")
    parser.add_argument("--single-pass", action="store_true", help="Enable single-pass mode for debugging (stops on EOS)")
    parser.add_argument("--startup-delay", type=float, default=0.3, help="Startup delay in seconds before sending initial stream (default: 0.3s)")
    args = parser.parse_args()

    loop_mode = not args.single_pass
    sender = SynchronizedStreamSender(args.config, loop=loop_mode)

    def signal_handler(sig, frame):
        sender.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sender.start(startup_delay=args.startup_delay)

if __name__ == "__main__":
    main()