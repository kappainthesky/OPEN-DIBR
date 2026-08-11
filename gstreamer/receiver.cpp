#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>
#include <csignal>
#include <iomanip>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Global shutdown flag and main loop pointer
static std::atomic<bool> g_running{true};
static GMainLoop* g_main_loop = nullptr;

struct StreamStats {
    std::string stream_name;
    std::atomic<uint64_t> frame_count;
    std::atomic<uint64_t> total_bytes;

    StreamStats(const std::string& name) : stream_name(name), frame_count(0), total_bytes(0) {}
};

static StreamStats g_rgb_stats("RGB Stream");
static StreamStats g_depth_stats("Depth Stream");

void signal_handler(int signum) {
    std::cout << "\n[Receiver] Signal " << signum << " received, stopping..." << std::endl;
    g_running = false;
    if (g_main_loop && g_main_loop_is_running(g_main_loop)) {
        g_main_loop_quit(g_main_loop);
    }
}

// Callback when appsink receives a new compressed HEVC sample
static GstFlowReturn on_new_sample(GstAppSink* appsink, gpointer user_data) {
    StreamStats* stats = static_cast<StreamStats*>(user_data);
    
    // Pull sample from appsink
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) {
        return GST_FLOW_ERROR;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (buffer) {
        GstMapInfo map_info;
        if (gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
            uint64_t current_frame = ++stats->frame_count;
            stats->total_bytes += map_info.size;

            GstClockTime pts = GST_BUFFER_PTS(buffer);
            double pts_ms = (GST_CLOCK_TIME_IS_VALID(pts)) ? (pts / 1000000.0) : 0.0;

            // Extract first few bytes (NAL unit start code and header)
            std::ostringstream hex_preview;
            size_t preview_len = std::min(map_info.size, static_cast<gsize>(8));
            for (size_t i = 0; i < preview_len; ++i) {
                hex_preview << std::hex << std::setw(2) << std::setfill('0') 
                            << static_cast<int>(map_info.data[i]) << " ";
            }

            // Print details for compressed HEVC buffer
            std::cout << "[" << stats->stream_name << "] Frame #" << std::setw(4) << current_frame
                      << " | Size: " << std::setw(7) << map_info.size << " bytes"
                      << " | PTS: " << std::fixed << std::setprecision(2) << std::setw(8) << pts_ms << " ms"
                      << " | Header: " << hex_preview.str() << std::endl;

            gst_buffer_unmap(buffer, &map_info);
        }
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

int main(int argc, char* argv[]) {
    std::string config_path = "camera_config.json";
    if (argc > 1) {
        config_path = argv[1];
    }

    std::cout << "==========================================================================" << std::endl;
    std::cout << "OpenDIBR GStreamer Receiver PoC - Compressed HEVC Buffer Sink" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    // Load camera startup configuration file
    std::ifstream cfg_file(config_path);
    if (!cfg_file.is_open()) {
        std::cerr << "[Receiver] Error: Failed to open configuration file: " << config_path << std::endl;
        return 1;
    }

    json config;
    try {
        cfg_file >> config;
    } catch (const std::exception& e) {
        std::cerr << "[Receiver] Error parsing JSON configuration: " << e.what() << std::endl;
        return 1;
    }

    // Extract network & camera settings
    int color_port = 5000;
    int depth_port = 5001;
    std::string host = "127.0.0.1";

    if (config.contains("streaming")) {
        color_port = config["streaming"].value("color_port", 5000);
        depth_port = config["streaming"].value("depth_port", 5001);
        host = config["streaming"].value("host", "127.0.0.1");
    }

    std::cout << "[Receiver] Config Loaded: " << config_path << std::endl;
    std::cout << "[Receiver] Target Host: " << host << std::endl;
    std::cout << "[Receiver] RGB UDP Port: " << color_port << std::endl;
    std::cout << "[Receiver] Depth UDP Port: " << depth_port << std::endl;

    if (config.contains("cameras") && !config["cameras"].empty()) {
        const auto& cam = config["cameras"][0];
        std::cout << "[Receiver] Camera Configuration:" << std::endl;
        if (cam.contains("Resolution")) {
            std::cout << "  - Resolution: " << cam["Resolution"][0] << "x" << cam["Resolution"][1] << std::endl;
        }
        if (cam.contains("Depth_range")) {
            std::cout << "  - Depth Range: [" << cam["Depth_range"][0] << ", " << cam["Depth_range"][1] << "]" << std::endl;
        }
        if (cam.contains("BitDepthColor") && cam.contains("BitDepthDepth")) {
            std::cout << "  - Bit Depth (Color / Depth): " << cam["BitDepthColor"] << " bit / " << cam["BitDepthDepth"] << " bit" << std::endl;
        }
    }

    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Build pipelines for RGB and Depth streams using appsink
    std::string rgb_pipeline_str = 
        "tcpclientsrc host=" + host + " port=" + std::to_string(color_port) + " timeout=5 ! "
        "h265parse config-interval=1 ! "
        "appsink name=rgb_appsink caps=\"video/x-h265, stream-format=byte-stream, alignment=au\" emit-signals=true";

    std::string depth_pipeline_str = 
        "tcpclientsrc host=" + host + " port=" + std::to_string(depth_port) + " timeout=5 ! "
        "h265parse config-interval=1 ! "
        "appsink name=depth_appsink caps=\"video/x-h265, stream-format=byte-stream, alignment=au\" emit-signals=true";

    GError* error = nullptr;
    GstElement* rgb_pipeline = gst_parse_launch(rgb_pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "[Receiver] RGB Pipeline Error: " << error->message << std::endl;
        g_clear_error(&error);
        return 1;
    }

    GstElement* depth_pipeline = gst_parse_launch(depth_pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "[Receiver] Depth Pipeline Error: " << error->message << std::endl;
        g_clear_error(&error);
        gst_object_unref(rgb_pipeline);
        return 1;
    }

    // Get appsink elements and connect callbacks
    GstElement* rgb_appsink = gst_bin_get_by_name(GST_BIN(rgb_pipeline), "rgb_appsink");
    GstElement* depth_appsink = gst_bin_get_by_name(GST_BIN(depth_pipeline), "depth_appsink");

    if (!rgb_appsink || !depth_appsink) {
        std::cerr << "[Receiver] Error: Failed to retrieve appsink elements from pipelines." << std::endl;
        return 1;
    }

    g_signal_connect(rgb_appsink, "new-sample", G_CALLBACK(on_new_sample), &g_rgb_stats);
    g_signal_connect(depth_appsink, "new-sample", G_CALLBACK(on_new_sample), &g_depth_stats);

    // Start pipelines
    std::cout << "[Receiver] Starting GStreamer appsink receiver pipelines..." << std::endl;
    gst_element_set_state(rgb_pipeline, GST_STATE_PLAYING);
    gst_element_set_state(depth_pipeline, GST_STATE_PLAYING);

    g_main_loop = g_main_loop_new(nullptr, FALSE);
    std::cout << "[Receiver] Listening for incoming compressed HEVC buffers (Press Ctrl+C to exit)..." << std::endl;
    g_main_loop_run(g_main_loop);

    // Cleanup
    std::cout << "\n[Receiver] Stopping receiver pipelines..." << std::endl;
    gst_element_set_state(rgb_pipeline, GST_STATE_NULL);
    gst_element_set_state(depth_pipeline, GST_STATE_NULL);

    gst_object_unref(rgb_appsink);
    gst_object_unref(depth_appsink);
    gst_object_unref(rgb_pipeline);
    gst_object_unref(depth_pipeline);

    if (g_main_loop) {
        g_main_loop_unref(g_main_loop);
    }

    std::cout << "==========================================================================" << std::endl;
    std::cout << "Receiver Summary Statistics:" << std::endl;
    std::cout << "  - RGB Stream Received:   " << g_rgb_stats.frame_count << " compressed HEVC buffers (" 
              << g_rgb_stats.total_bytes << " bytes)" << std::endl;
    std::cout << "  - Depth Stream Received: " << g_depth_stats.frame_count << " compressed HEVC buffers (" 
              << g_depth_stats.total_bytes << " bytes)" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    return 0;
}
