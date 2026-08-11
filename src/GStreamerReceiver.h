#ifndef GSTREAMER_RECEIVER_H
#define GSTREAMER_RECEIVER_H

/*
 * Fixed version of GStreamerReceiver.
 *
 * Changes vs original:
 * 1. rtpjitterbuffer: drop-on-latency=false (was true) and latency raised
 *    to 200ms (was 50ms). The original settings would silently drop RTP
 *    packets under any jitter, which combined with the sender's changing
 *    SSRC on loop restart (see sender fix) or normal network jitter, could
 *    desync color vs depth permanently (once one stream loses a frame and
 *    the other doesn't, every subsequent pairing is off by one frame).
 * 2. Internal queue: max-size-buffers raised from 10 -> 100 and leaky
 *    behavior removed (was "leaky=downstream", silently dropping the
 *    oldest buffered AU under backpressure). Default queue behavior blocks
 *    upstream instead of dropping, which is safer for a fixed-rate
 *    "must not lose a frame" pipeline like this one.
 * 3. Added PeekFrontPTS() / PopFront() plus a free function
 *    GetSynchronizedPair() so a caller managing two GStreamerReceiver
 *    instances (color + depth) can explicitly pair frames by PTS instead
 *    of blindly calling Demux() on each and assuming they stay aligned.
 *    This is the fix for the "no color/depth PTS matching" issue: even
 *    with a healthier pipeline, one dropped/late packet anywhere upstream
 *    (network, jitterbuffer, queue) can otherwise misalign the two streams
 *    permanently.
 *
 * Usage for synchronized reads (recommended for openDIBR):
 *
 *   GStreamerReceiver colorRx(5000, 96, "Color");
 *   GStreamerReceiver depthRx(5001, 97, "Depth");
 *   HEVCFrame colorFrame, depthFrame;
 *   while (running) {
 *       if (GetSynchronizedPair(colorRx, depthRx, colorFrame, depthFrame, 5.0)) {
 *           // feed colorFrame.data / depthFrame.data to the decoder pair
 *       }
 *   }
 *
 * Demux() is kept for backward compatibility (unsynchronized, FIFO order),
 * but for openDIBR you should prefer GetSynchronizedPair().
 */

#include <iostream>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include "FFmpegDemuxer.h"

struct HEVCFrame {
    std::vector<uint8_t> data;
    int64_t pts = -1; // in nanoseconds
    int64_t dts = -1; // in nanoseconds
    bool is_keyframe = false;
    bool is_idr = false;
    bool is_discont = false;
    int first_vcl_nal_type = -1;
};

// Helper function to inspect HEVC Annex-B NAL unit headers for VCL NAL type & IDR status
inline void inspectHEVCFrame(const uint8_t* data, size_t size, int& vcl_nal_type, bool& is_idr) {
    vcl_nal_type = -1;
    is_idr = false;
    size_t i = 0;
    while (i + 4 < size) {
        if (data[i] == 0 && data[i+1] == 0) {
            size_t nal_offset = 0;
            if (data[i+2] == 1) {
                nal_offset = i + 3;
            } else if (data[i+2] == 0 && data[i+3] == 1) {
                nal_offset = i + 4;
            }
            if (nal_offset > 0 && nal_offset < size) {
                uint8_t nal_header = data[nal_offset];
                int type = (nal_header >> 1) & 0x3F;
                if (type <= 31) { // VCL NAL unit (Slice)
                    if (vcl_nal_type == -1) {
                        vcl_nal_type = type;
                        if (type >= 16 && type <= 21) { // HEVC IRAP / IDR NAL types
                            is_idr = true;
                        }
                    }
                }
            }
        }
        i++;
    }
}

class GStreamerReceiver : public IDemuxer {
private:
    std::string name;
    int port;
    int payload_type;
    std::string host;
    GstElement* pipeline = nullptr;
    GstElement* appsink = nullptr;
    GMainLoop* main_loop = nullptr;
    std::thread loop_thread;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<HEVCFrame> frame_queue;
    size_t max_queue_size = 300;

    HEVCFrame current_frame;
    std::atomic<bool> is_running{false};

    // Metrics & Diagnostic Counters
    std::atomic<uint64_t> appsink_au_count{0};
    std::atomic<uint64_t> demux_calls_count{0};
    std::atomic<uint64_t> timeout_count{0};
    std::atomic<uint64_t> discont_count{0};
    std::atomic<uint64_t> idr_count{0};
    std::atomic<uint64_t> dropped_frames_count{0};

    int64_t base_pts = -1;
    double last_pts_ms = -1.0;
    double total_pts_delta_ms = 0.0;
    uint64_t pts_delta_count = 0;

    std::atomic<bool> has_received_first_idr{false};

    // Cross-stream sync: pointer to the paired receiver (color<->depth).
    // Set via SetPeer() once, right after both receivers are constructed.
    // When set, Demux() self-corrects against the peer's queue before
    // returning a frame, so Pool.h (and any other IDemuxer consumer)
    // needs ZERO changes -- the Demux() signature/contract is unchanged.
    GStreamerReceiver* peer = nullptr;
    double sync_tolerance_ms = 5.0;

    static GstFlowReturn on_new_sample_cb(GstAppSink* sink, gpointer user_data) {
        GStreamerReceiver* self = static_cast<GStreamerReceiver*>(user_data);
        return self->handle_new_sample(sink);
    }

    GstFlowReturn handle_new_sample(GstAppSink* sink) {
        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample) return GST_FLOW_ERROR;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (buffer) {
            GstMapInfo map_info;
            if (gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
                HEVCFrame frame;
                frame.data.assign(map_info.data, map_info.data + map_info.size);

                GstClockTime raw_pts = GST_BUFFER_PTS(buffer);
                GstClockTime raw_dts = GST_BUFFER_DTS(buffer);

                if (GST_CLOCK_TIME_IS_VALID(raw_pts)) {
                    int64_t pts_val = static_cast<int64_t>(raw_pts);
                    if (base_pts < 0) {
                        base_pts = pts_val;
                    }
                    frame.pts = pts_val - base_pts;
                } else {
                    frame.pts = -1;
                }

                frame.dts = GST_CLOCK_TIME_IS_VALID(raw_dts) ? (static_cast<int64_t>(raw_dts) - (base_pts >= 0 ? base_pts : 0)) : -1;

                frame.is_discont = GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DISCONT);
                if (frame.is_discont) {
                    discont_count++;
                }

                // Detailed NAL Unit & Keyframe Inspection
                inspectHEVCFrame(frame.data.data(), frame.data.size(), frame.first_vcl_nal_type, frame.is_idr);
                frame.is_keyframe = frame.is_idr;
                if (frame.is_idr) {
                    idr_count++;
                }

                gst_buffer_unmap(buffer, &map_info);

                uint64_t count = ++appsink_au_count;
                double pts_ms = (frame.pts >= 0) ? (frame.pts / 1000000.0) : -1.0;
                double dts_ms = (frame.dts >= 0) ? (frame.dts / 1000000.0) : -1.0;

                double delta_pts = 0.0;
                if (last_pts_ms >= 0.0 && pts_ms >= 0.0) {
                    delta_pts = pts_ms - last_pts_ms;
                    total_pts_delta_ms += delta_pts;
                    pts_delta_count++;
                }
                if (pts_ms >= 0.0) {
                    last_pts_ms = pts_ms;
                }

                // Log details for first 10 frames or on DISCONT
                if (count <= 10 || frame.is_discont) {
                    std::ostringstream hex_hdr;
                    size_t preview_len = std::min(frame.data.size(), static_cast<size_t>(16));
                    for (size_t i = 0; i < preview_len; ++i) {
                        hex_hdr << std::hex << std::setw(2) << std::setfill('0')
                                << static_cast<int>(frame.data[i]) << " ";
                    }

                    std::cout << "[" << name << " Port " << port << " pt=" << payload_type << "] AU #" << std::setw(3) << count
                              << " | Size: " << std::setw(7) << frame.data.size() << " B"
                              << " | PTS: " << std::fixed << std::setprecision(2) << std::setw(7) << pts_ms << " ms"
                              << " | dPTS: " << std::setw(6) << delta_pts << " ms"
                              << " | VCL NAL: " << std::setw(2) << frame.first_vcl_nal_type
                              << " | IDR: " << (frame.is_keyframe ? "YES" : " NO")
                              << " | DISCONT: " << (frame.is_discont ? "YES" : " NO")
                              << " | First 16B: " << hex_hdr.str() << std::endl;

                    if (count == 1) {
                        std::cout << "[" << name << "] AU #1 NAL list: ";
                        size_t idx = 0;
                        while (idx + 4 < frame.data.size()) {
                            if (frame.data[idx] == 0 && frame.data[idx+1] == 0) {
                                size_t off = 0;
                                if (frame.data[idx+2] == 1) off = idx + 3;
                                else if (frame.data[idx+2] == 0 && frame.data[idx+3] == 1) off = idx + 4;
                                if (off > 0 && off < frame.data.size()) {
                                    int nal_t = (frame.data[off] >> 1) & 0x3F;
                                    std::cout << nal_t << " ";
                                }
                            }
                            idx++;
                        }
                        std::cout << std::endl;
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (frame_queue.size() >= max_queue_size) {
                        // Drop oldest, but count it so it is visible in
                        // diagnostics instead of failing silently.
                        frame_queue.pop_front();
                        dropped_frames_count++;
                    }
                    frame_queue.push_back(std::move(frame));
                }
                queue_cv.notify_one();
            }
        }
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

public:
    GStreamerReceiver(int port, int payload_type = 96, const std::string& name = "GStreamerReceiver", const std::string& host = "127.0.0.1")
        : port(port), payload_type(payload_type), name(name), host(host) {

        static std::once_flag gst_init_flag;
        std::call_once(gst_init_flag, []() {
            gst_init(nullptr, nullptr);
        });

        std::string pipeline_str =
            "tcpclientsrc host=" + host + " port=" + std::to_string(port) + " timeout=5 ! "
            "h265parse config-interval=1 ! "
            "queue max-size-buffers=100 ! "
            "appsink name=sink caps=\"video/x-h265, stream-format=byte-stream, alignment=au\" emit-signals=true sync=false";

        int max_attempts = 5;
        bool success = false;
        for (int attempt = 1; attempt <= max_attempts; ++attempt) {
            GError* error = nullptr;
            pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
            if (error) {
                std::cerr << "[" << name << "] GStreamer pipeline parse error: " << error->message << std::endl;
                g_clear_error(&error);
                return;
            }

            appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
            if (!appsink) {
                std::cerr << "[" << name << "] Failed to find appsink in pipeline." << std::endl;
                gst_object_unref(pipeline);
                pipeline = nullptr;
                return;
            }

            g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample_cb), this);

            is_running = true;
            main_loop = g_main_loop_new(nullptr, FALSE);

            GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
            if (ret != GST_STATE_CHANGE_FAILURE) {
                success = true;
                break;
            }

            std::cerr << "[" << name << "] TCP connect attempt " << attempt << "/" << max_attempts
                      << " to " << host << ":" << port << " failed. Retrying in 1s..." << std::endl;

            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(appsink);
            gst_object_unref(pipeline);
            g_main_loop_unref(main_loop);
            pipeline = nullptr;
            appsink = nullptr;
            main_loop = nullptr;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (!success) {
            std::cerr << "[" << name << "] Critical Error: Could not connect TCP client to "
                      << host << ":" << port << " after " << max_attempts << " attempts. Make sure stream_sender.py (TCP server) is running first!" << std::endl;
            return;
        }

        loop_thread = std::thread([this]() {
            g_main_loop_run(main_loop);
        });

        std::cout << "[" << name << "] TCP Receiver connected to " << host << ":" << port
                  << " (timeout=5s, alignment=au)" << std::endl;
    }

    virtual ~GStreamerReceiver() override {
        is_running = false;
        queue_cv.notify_all();

        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        if (main_loop && g_main_loop_is_running(main_loop)) {
            g_main_loop_quit(main_loop);
        }
        if (loop_thread.joinable()) {
            loop_thread.join();
        }
        if (appsink) {
            gst_object_unref(appsink);
        }
        if (pipeline) {
            gst_object_unref(pipeline);
        }
        if (main_loop) {
            g_main_loop_unref(main_loop);
        }
    }

    // Call once after constructing both the color and depth receivers, e.g.:
    //   colorRx.SetPeer(&depthRx);
    //   depthRx.SetPeer(&colorRx);
    // Wherever your demuxers vector is built for --gstreamer-input mode.
    // No changes to Pool.h are needed -- Demux() below handles the rest.
    void SetPeer(GStreamerReceiver* p, double tolerance_ms = 5.0) {
        peer = p;
        sync_tolerance_ms = tolerance_ms;
    }

    // Demux() now self-corrects against its peer (if set) before returning
    // a frame, so color and depth stay PTS-aligned even if one stream
    // drops a packet somewhere upstream. Pool.h calls this exactly as
    // before -- signature and return contract are unchanged.
    bool Demux(uint8_t **ppVideo, int *pnVideoBytes, bool *pbLooped = nullptr) override {
        if (pbLooped) *pbLooped = false;
        demux_calls_count++;

        // 1. Ensure we do not return any frame until we hit our FIRST IDR keyframe!
        if (!has_received_first_idr) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            while (is_running) {
                while (!frame_queue.empty()) {
                    if (frame_queue.front().is_idr) {
                        has_received_first_idr = true;
                        if (pbLooped) *pbLooped = true; // Signal decoder DPB flush on initial IDR / stream sync
                        break;
                    }
                    // Drop pre-IDR P-frames
                    frame_queue.pop_front();
                }
                if (has_received_first_idr) break;
                if (queue_cv.wait_for(lock, std::chrono::milliseconds(2000)) == std::cv_status::timeout) {
                    if (!is_running) break;
                }
            }
        }

        if (peer) {
            // Bounded retries: avoids ever looping forever if something
            // upstream stalls; falls through to normal FIFO behavior below
            // if alignment can't be confirmed within the attempt budget.
            for (int attempt = 0; attempt < 50; ++attempt) {
                double ownPtsMs, peerPtsMs;
                if (!PeekFrontPTSMs(ownPtsMs, std::chrono::milliseconds(500))) break;
                if (!peer->PeekFrontPTSMs(peerPtsMs, std::chrono::milliseconds(500))) break;

                double diff = ownPtsMs - peerPtsMs;
                if (std::fabs(diff) <= sync_tolerance_ms) break; // aligned enough

                if (diff < -sync_tolerance_ms) {
                    // We are behind peer -> drop our own stale frame and retry.
                    DropFront();
                } else {
                    // We are ahead of peer -> do not mutate peer's queue.
                    // Peer will drop its own stale frame when it calls Demux().
                    break;
                }
            }
        }

        std::unique_lock<std::mutex> lock(queue_mutex);
        while (frame_queue.empty() && is_running) {
            if (queue_cv.wait_for(lock, std::chrono::milliseconds(2000)) == std::cv_status::timeout) {
                timeout_count++;
                std::cerr << "[" << name << "] Warning: Timeout (2000ms) waiting for frame from TCP sender " << host << ":" << port << std::endl;
                if (!is_running) break;
            }
        }

        if (frame_queue.empty() || !is_running) {
            *ppVideo = nullptr;
            *pnVideoBytes = 0;
            return false;
        }

        // Handle discontinuity (e.g. stream restart or packet loss boundary)
        if (frame_queue.front().is_discont) {
            if (!frame_queue.front().is_idr) {
                // Discontinuity hit on non-IDR frame -> wait for clean keyframe
                has_received_first_idr = false;
                lock.unlock();
                return Demux(ppVideo, pnVideoBytes, pbLooped);
            } else {
                if (pbLooped) *pbLooped = true;
            }
        }

        current_frame = std::move(frame_queue.front());
        frame_queue.pop_front();
        lock.unlock();

        *ppVideo = current_frame.data.data();
        *pnVideoBytes = static_cast<int>(current_frame.data.size());
        return true;
    }

    // Peek the PTS (ms) of the frame at the front of the queue without
    // removing it. Returns false if the queue is empty (caller should wait
    // and retry). Used by GetSynchronizedPair() below.
    bool PeekFrontPTSMs(double& pts_ms_out, std::chrono::milliseconds wait_for = std::chrono::milliseconds(2000)) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (frame_queue.empty() && is_running) {
            queue_cv.wait_for(lock, wait_for, [this] { return !frame_queue.empty() || !is_running; });
        }
        if (frame_queue.empty()) return false;
        pts_ms_out = (frame_queue.front().pts >= 0) ? (frame_queue.front().pts / 1000000.0) : -1.0;
        return true;
    }

    // Pop the frame currently at the front of the queue into out_frame.
    // Returns false if the queue is empty.
    bool PopFront(HEVCFrame& out_frame) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (frame_queue.empty()) return false;
        out_frame = std::move(frame_queue.front());
        frame_queue.pop_front();
        return true;
    }

    // Drop the frame at the front of the queue (used when this stream is
    // "ahead" of its paired stream and needs to catch down).
    bool DropFront() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (frame_queue.empty()) return false;
        frame_queue.pop_front();
        dropped_frames_count++;
        return true;
    }

    AVCodecID GetVideoCodec() override {
        return AV_CODEC_ID_HEVC;
    }

    uint64_t getAppsinkAUCount() const { return appsink_au_count.load(); }
    uint64_t getDemuxCallsCount() const { return demux_calls_count.load(); }
    uint64_t getTimeoutCount() const { return timeout_count.load(); }
    uint64_t getDiscontCount() const { return discont_count.load(); }
    uint64_t getIDRCount() const { return idr_count.load(); }
    uint64_t getDroppedFramesCount() const { return dropped_frames_count.load(); }
    double getAvgPTSDelta() const {
        return pts_delta_count > 0 ? (total_pts_delta_ms / pts_delta_count) : 0.0;
    }
    int64_t getCurrentPTS() const { return current_frame.pts; }
};

// Pulls one frame from each of the two receivers such that their PTS values
// are within tolerance_ms of each other. If one stream is ahead of the
// other (e.g. because a packet was dropped somewhere upstream on the
// other stream), the ahead stream's front frame is held while the lagging
// stream catches up, or the lagging stream's frame is dropped if it can
// never catch up (PTS jump). This prevents the "permanent one-frame
// offset" failure mode that silently ruins color/depth correspondence in
// openDIBR.
inline bool GetSynchronizedPair(GStreamerReceiver& colorRx, GStreamerReceiver& depthRx,
                                 HEVCFrame& outColor, HEVCFrame& outDepth,
                                 double tolerance_ms = 5.0,
                                 int max_resync_attempts = 30) {
    for (int attempt = 0; attempt < max_resync_attempts; ++attempt) {
        double colorPts, depthPts;
        if (!colorRx.PeekFrontPTSMs(colorPts)) return false;
        if (!depthRx.PeekFrontPTSMs(depthPts)) return false;

        double diff = colorPts - depthPts;

        if (std::fabs(diff) <= tolerance_ms) {
            bool okC = colorRx.PopFront(outColor);
            bool okD = depthRx.PopFront(outDepth);
            return okC && okD;
        }

        // Color is ahead of depth -> drop depth's stale frame and retry.
        if (diff > tolerance_ms) {
            depthRx.DropFront();
        } else {
            // Depth is ahead of color -> drop color's stale frame and retry.
            colorRx.DropFront();
        }
    }
    // Could not align within the attempt budget; give up this cycle so the
    // caller can log/skip rather than pairing mismatched frames.
    return false;
}

#endif // GSTREAMER_RECEIVER_H