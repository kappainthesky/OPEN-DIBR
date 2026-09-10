#ifndef REALSENSE_RECEIVER_H
#define REALSENSE_RECEIVER_H

#include <librealsense2/rs.hpp>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "ioHelper.h"

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

struct RealSenseIntrinsics {
    int width = 640;
    int height = 480;
    float fx = 0.0f;
    float fy = 0.0f;
    float ppx = 0.0f;
    float ppy = 0.0f;
    float depth_scale = 0.001f;
};

struct RealSenseFilterConfig {
    bool enableThreshold = true;
    float minDistance = 0.2f;
    float maxDistance = 5.0f;

    bool enableDisparity = true; // Disparity-domain filtering (Intel recommended for stereo depth)

    bool enableSpatial = true;
    float spatialAlpha = 0.5f;
    float spatialDelta = 20.0f;
    int spatialMagnitude = 2;
    int spatialHoleFill = 2; // 2: 3x3 hole fill

    bool enableTemporal = true;
    float temporalAlpha = 0.4f;
    float temporalDelta = 20.0f;
    int temporalPersistency = 3;

    bool enableHoleFilling = true;
    int holeFillingMode = 2; // 2: nearest neighbor from around (preserves solid foreground)

    bool enableDiagnostics = false;
    bool enableRgbBridge = true; // Stream RGB frames to eye tracker on 127.0.0.1:9998
    int rgbBridgePort = 9998;
};

// High-speed non-blocking loopback TCP server to share live RealSense RGB frames with Python eye tracker
class RgbFrameServer {
private:
    int m_serverSock = -1;
    int m_clientSock = -1;
    std::thread m_thread;
    std::atomic<bool> m_running{false};
    std::mutex m_frameMutex;
    std::condition_variable m_cv;
    std::vector<uint8_t> m_latestRgb;
    bool m_hasNewFrame = false;

    void ServerLoop() {
        while (m_running) {
            if (m_clientSock == -1) {
                sockaddr_in clientAddr{};
                socklen_t clientLen = sizeof(clientAddr);
                int client = accept(m_serverSock, (sockaddr*)&clientAddr, &clientLen);
                if (client >= 0) {
                    int flag = 1;
                    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag));
                    m_clientSock = client;
                    std::cout << "[EyeTracker Bridge] Python eye tracker connected on TCP port 9998." << std::endl;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
            }

            std::vector<uint8_t> frameCopy;
            {
                std::unique_lock<std::mutex> lock(m_frameMutex);
                m_cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return m_hasNewFrame || !m_running; });
                if (!m_running) break;
                if (!m_hasNewFrame) continue;
                frameCopy = m_latestRgb;
                m_hasNewFrame = false;
            }

            if (m_clientSock >= 0 && !frameCopy.empty()) {
#ifndef _WIN32
                ssize_t sent = send(m_clientSock, frameCopy.data(), frameCopy.size(), MSG_NOSIGNAL);
#else
                int sent = send(m_clientSock, (const char*)frameCopy.data(), (int)frameCopy.size(), 0);
#endif
                if (sent <= 0) {
#ifndef _WIN32
                    close(m_clientSock);
#else
                    closesocket(m_clientSock);
#endif
                    m_clientSock = -1;
                    std::cout << "[EyeTracker Bridge] Eye tracker client disconnected." << std::endl;
                }
            }
        }
    }

public:
    RgbFrameServer() = default;
    ~RgbFrameServer() { Stop(); }

    bool Start(int port = 9998) {
        Stop();
        m_serverSock = socket(AF_INET, SOCK_STREAM, 0);
        if (m_serverSock < 0) return false;

        int opt = 1;
        setsockopt(m_serverSock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(port);

        if (bind(m_serverSock, (sockaddr*)&addr, sizeof(addr)) < 0) {
#ifndef _WIN32
            close(m_serverSock);
#else
            closesocket(m_serverSock);
#endif
            m_serverSock = -1;
            return false;
        }

        if (listen(m_serverSock, 1) < 0) {
#ifndef _WIN32
            close(m_serverSock);
#else
            closesocket(m_serverSock);
#endif
            m_serverSock = -1;
            return false;
        }

        m_running = true;
        m_thread = std::thread(&RgbFrameServer::ServerLoop, this);
        std::cout << "[EyeTracker Bridge] RealSense RGB Frame Server active on 127.0.0.1:" << port << std::endl;
        return true;
    }

    void PushFrame(const uint8_t* rgbData, size_t sizeBytes) {
        if (!m_running || m_clientSock == -1) return;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            if (m_latestRgb.size() != sizeBytes) {
                m_latestRgb.resize(sizeBytes);
            }
            std::memcpy(m_latestRgb.data(), rgbData, sizeBytes);
            m_hasNewFrame = true;
        }
        m_cv.notify_one();
    }

    void PushRgbdFrame(const uint8_t* rgbData, const uint16_t* depthData, size_t rgbBytes, size_t depthBytes) {
        if (!m_running || m_clientSock == -1) return;
        size_t totalBytes = rgbBytes + depthBytes;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            if (m_latestRgb.size() != totalBytes) {
                m_latestRgb.resize(totalBytes);
            }
            if (rgbData && rgbBytes > 0) {
                std::memcpy(m_latestRgb.data(), rgbData, rgbBytes);
            }
            if (depthData && depthBytes > 0) {
                std::memcpy(m_latestRgb.data() + rgbBytes, depthData, depthBytes);
            }
            m_hasNewFrame = true;
        }
        m_cv.notify_one();
    }

    void PushStereoFrame(const uint8_t* leftRgb, const uint8_t* rightRgb, int width, int height) {
        if (!m_running || m_clientSock == -1) return;
        size_t singleFrameBytes = static_cast<size_t>(width * height * 3);
        size_t totalBytes = singleFrameBytes * 2;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            if (m_latestRgb.size() != totalBytes) {
                m_latestRgb.resize(totalBytes);
            }
            if (leftRgb && rightRgb) {
                std::memcpy(m_latestRgb.data(), leftRgb, singleFrameBytes);
                std::memcpy(m_latestRgb.data() + singleFrameBytes, rightRgb, singleFrameBytes);
            } else if (leftRgb) {
                std::memcpy(m_latestRgb.data(), leftRgb, singleFrameBytes);
                std::memcpy(m_latestRgb.data() + singleFrameBytes, leftRgb, singleFrameBytes);
            }
            m_hasNewFrame = true;
        }
        m_cv.notify_one();
    }

    void Stop() {
        m_running = false;
        m_cv.notify_all();
        if (m_clientSock >= 0) {
#ifndef _WIN32
            close(m_clientSock);
#else
            closesocket(m_clientSock);
#endif
            m_clientSock = -1;
        }
        if (m_serverSock >= 0) {
#ifndef _WIN32
            close(m_serverSock);
#else
            closesocket(m_serverSock);
#endif
            m_serverSock = -1;
        }
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
};

// Represents a single physical RealSense Camera device (e.g. D455 #1, D455 #2)
class RealSenseDevice {
private:
    std::string m_serialNumber;
    std::string m_role;
    std::string m_modelName;

    rs2::pipeline m_pipe;
    rs2::config m_cfg;
    std::unique_ptr<rs2::align> m_align;
    bool m_isStreaming = false;

    // Filters
    rs2::threshold_filter m_thresholdFilter;
    rs2::disparity_transform m_depthToDisparity{true};
    rs2::spatial_filter m_spatialFilter;
    rs2::temporal_filter m_temporalFilter;
    rs2::disparity_transform m_disparityToDepth{false};
    rs2::hole_filling_filter m_holeFillingFilter;

    int m_width = 640;
    int m_height = 480;
    int m_fps = 30;
    float m_zNear = 0.2f;
    float m_zFar = 5.0f;
    float m_depthScale = 0.001f;

    RealSenseIntrinsics m_intrinsics;
    RealSenseFilterConfig m_filterConfig;
    std::vector<uint16_t> m_depthLut;

    std::vector<uint8_t> m_rgbBuffer;
    std::vector<uint16_t> m_normalizedDepthBuffer;
    std::vector<uint16_t> m_rawDepthBuffer;

    uint64_t m_frameCount = 0;
    double m_lastColorTimestamp = 0.0;
    double m_lastDepthTimestamp = 0.0;

    void BuildDepthLut() {
        m_depthLut.assign(65536, 0);
        float inv_near = 1.0f / m_zNear;
        float inv_far = 1.0f / m_zFar;
        float inv_range = inv_near - inv_far;

        for (int raw = 0; raw < 65536; ++raw) {
            float z_meters = raw * m_depthScale;
            if (raw == 0 || z_meters < m_zNear || z_meters > m_zFar) {
                m_depthLut[raw] = 0;
            } else {
                float inv_z = 1.0f / z_meters;
                float d = (inv_z - inv_far) / inv_range;
                d = std::max(0.0f, std::min(1.0f, d));
                m_depthLut[raw] = static_cast<uint16_t>(d * 65535.0f + 0.5f);
            }
        }
    }

    void ConfigureFilters() {
        if (m_filterConfig.enableThreshold) {
            m_thresholdFilter.set_option(RS2_OPTION_MIN_DISTANCE, m_filterConfig.minDistance);
            m_thresholdFilter.set_option(RS2_OPTION_MAX_DISTANCE, m_filterConfig.maxDistance);
        }
        if (m_filterConfig.enableSpatial) {
            m_spatialFilter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, m_filterConfig.spatialAlpha);
            m_spatialFilter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, m_filterConfig.spatialDelta);
            m_spatialFilter.set_option(RS2_OPTION_FILTER_MAGNITUDE, static_cast<float>(m_filterConfig.spatialMagnitude));
            m_spatialFilter.set_option(RS2_OPTION_HOLES_FILL, static_cast<float>(m_filterConfig.spatialHoleFill));
        }
        if (m_filterConfig.enableTemporal) {
            m_temporalFilter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, m_filterConfig.temporalAlpha);
            m_temporalFilter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, m_filterConfig.temporalDelta);
            m_temporalFilter.set_option(RS2_OPTION_HOLES_FILL, static_cast<float>(m_filterConfig.temporalPersistency));
        }
        if (m_filterConfig.enableHoleFilling) {
            m_holeFillingFilter.set_option(RS2_OPTION_HOLES_FILL, static_cast<float>(m_filterConfig.holeFillingMode));
        }
    }

public:
    RealSenseDevice(const std::string& serial, const std::string& role, const std::string& model = "RealSense")
        : m_serialNumber(serial), m_role(role), m_modelName(model) {}

    ~RealSenseDevice() {
        Stop();
    }

    bool Start(int width, int height, int fps, float zNear, float zFar, const RealSenseFilterConfig& filterConfig) {
        m_width = width;
        m_height = height;
        m_fps = fps;
        m_zNear = zNear;
        m_zFar = zFar;
        m_filterConfig = filterConfig;
        m_filterConfig.minDistance = zNear;
        m_filterConfig.maxDistance = zFar;

        int maxRetries = 3;
        for (int attempt = 1; attempt <= maxRetries; ++attempt) {
            try {
                if (!m_serialNumber.empty()) {
                    m_cfg.enable_device(m_serialNumber);
                }
                m_cfg.enable_stream(RS2_STREAM_COLOR, m_width, m_height, RS2_FORMAT_RGB8, m_fps);
                m_cfg.enable_stream(RS2_STREAM_DEPTH, m_width, m_height, RS2_FORMAT_Z16, m_fps);

                rs2::pipeline_profile profile = m_pipe.start(m_cfg);
                rs2::device dev = profile.get_device();
                m_modelName = dev.get_info(RS2_CAMERA_INFO_NAME);
                m_serialNumber = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

                auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
                auto rs_intr = color_stream.get_intrinsics();
                m_intrinsics.width = rs_intr.width;
                m_intrinsics.height = rs_intr.height;
                m_intrinsics.fx = rs_intr.fx;
                m_intrinsics.fy = rs_intr.fy;
                m_intrinsics.ppx = rs_intr.ppx;
                m_intrinsics.ppy = rs_intr.ppy;

                rs2::depth_sensor ds = dev.first<rs2::depth_sensor>();
                m_depthScale = ds.get_depth_scale();
                m_intrinsics.depth_scale = m_depthScale;

                // Configure depth sensor hardware for minimum holes and maximum density
                if (ds.supports(RS2_OPTION_EMITTER_ENABLED)) {
                    try { ds.set_option(RS2_OPTION_EMITTER_ENABLED, 1.0f); } catch (...) {}
                }
                if (ds.supports(RS2_OPTION_VISUAL_PRESET)) {
                    try { ds.set_option(RS2_OPTION_VISUAL_PRESET, 4.0f); /* RS2_RS400_VISUAL_PRESET_HIGH_DENSITY */ } catch (...) {}
                }
                if (ds.supports(RS2_OPTION_LASER_POWER)) {
                    try {
                        auto range = ds.get_option_range(RS2_OPTION_LASER_POWER);
                        float power = std::min(range.max, std::max(range.min, 240.0f));
                        ds.set_option(RS2_OPTION_LASER_POWER, power);
                    } catch (...) {}
                }

                m_align = std::unique_ptr<rs2::align>(new rs2::align(RS2_STREAM_COLOR));

                ConfigureFilters();
                BuildDepthLut();

                m_rgbBuffer.resize(static_cast<size_t>(m_width * m_height * 3));
                m_normalizedDepthBuffer.resize(static_cast<size_t>(m_width * m_height));
                m_rawDepthBuffer.resize(static_cast<size_t>(m_width * m_height));

                m_isStreaming = true;
                m_frameCount = 0;

                // Warmup flush
                for (int wf = 0; wf < 5; ++wf) {
                    try {
                        m_pipe.wait_for_frames(4000);
                    } catch (...) {}
                }

                return true;
            } catch (const std::exception& e) {
                if (attempt < maxRetries) {
                    std::cerr << "[RealSense " << m_role << " (" << m_serialNumber << ")] Attempt " << attempt 
                              << " failed: " << e.what() << ". Retrying in 400ms..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(400));
                } else {
                    std::cerr << "[RealSense " << m_role << " (" << m_serialNumber << ")] Initialization failed: " << e.what() << std::endl;
                    m_isStreaming = false;
                    return false;
                }
            }
        }
        return false;
    }

    void Stop() {
        if (m_isStreaming) {
            try {
                m_pipe.stop();
            } catch (...) {}
            m_isStreaming = false;
        }
    }

    bool CaptureNextFrame(const uint8_t*& outRgb, const uint16_t*& outNormalizedDepth, unsigned int timeoutMs = 2000) {
        if (!m_isStreaming) return false;

        try {
            rs2::frameset frames = m_pipe.wait_for_frames(timeoutMs);
            rs2::frameset aligned_frames = m_align->process(frames);

            rs2::video_frame color_frame = aligned_frames.get_color_frame();
            rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();

            if (!color_frame || !depth_frame) {
                return false;
            }

            m_lastColorTimestamp = color_frame.get_timestamp();
            m_lastDepthTimestamp = depth_frame.get_timestamp();

            rs2::frame filtered_depth = depth_frame;
            if (m_filterConfig.enableThreshold) filtered_depth = m_thresholdFilter.process(filtered_depth);
            if (m_filterConfig.enableDisparity) filtered_depth = m_depthToDisparity.process(filtered_depth);
            if (m_filterConfig.enableSpatial) filtered_depth = m_spatialFilter.process(filtered_depth);
            if (m_filterConfig.enableTemporal) filtered_depth = m_temporalFilter.process(filtered_depth);
            if (m_filterConfig.enableDisparity) filtered_depth = m_disparityToDepth.process(filtered_depth);
            if (m_filterConfig.enableHoleFilling) filtered_depth = m_holeFillingFilter.process(filtered_depth);

            rs2::depth_frame processed_depth = filtered_depth.as<rs2::depth_frame>();

            const uint8_t* raw_rgb = static_cast<const uint8_t*>(color_frame.get_data());
            const uint16_t* raw_depth = static_cast<const uint16_t*>(processed_depth.get_data());

            size_t pixel_count = static_cast<size_t>(m_width * m_height);
            std::memcpy(m_rgbBuffer.data(), raw_rgb, pixel_count * 3);
            m_rawDepthBuffer.resize(pixel_count);
            std::memcpy(m_rawDepthBuffer.data(), raw_depth, pixel_count * sizeof(uint16_t));

            for (size_t i = 0; i < pixel_count; ++i) {
                uint16_t val = raw_depth[i];
                m_normalizedDepthBuffer[i] = m_depthLut[val];
            }

            m_frameCount++;
            outRgb = m_rgbBuffer.data();
            outNormalizedDepth = m_normalizedDepthBuffer.data();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[RealSense " << m_role << "] Frame capture timeout: " << e.what() << std::endl;
            return false;
        }
    }

    void UpdateFilterConfig(const RealSenseFilterConfig& cfg) {
        m_filterConfig = cfg;
        ConfigureFilters();
    }

    const std::string& GetSerialNumber() const { return m_serialNumber; }
    const std::string& GetRole() const { return m_role; }
    const std::string& GetModelName() const { return m_modelName; }
    const RealSenseIntrinsics& GetIntrinsics() const { return m_intrinsics; }
    const uint16_t* GetRawDepth() const { return m_rawDepthBuffer.data(); }
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    float GetZNear() const { return m_zNear; }
    float GetZFar() const { return m_zFar; }
    bool IsStreaming() const { return m_isStreaming; }
    uint64_t GetFrameCount() const { return m_frameCount; }
    double GetLastColorTimestamp() const { return m_lastColorTimestamp; }
    double GetLastDepthTimestamp() const { return m_lastDepthTimestamp; }
};

// Multi-device Manager: orchestrates multiple RealSense cameras and the RGB frame bridge
class RealSenseReceiver {
private:
    std::vector<std::unique_ptr<RealSenseDevice>> m_devices;
    RgbFrameServer m_rgbServer;
    RealSenseFilterConfig m_filterConfig;
    bool m_isStreaming = false;
    uint64_t m_globalFrameCount = 0;

public:
    RealSenseReceiver() = default;
    ~RealSenseReceiver() { Stop(); }

    void SetFilterConfig(const RealSenseFilterConfig& config) {
        m_filterConfig = config;
    }

    void UpdateFilterConfig(const RealSenseFilterConfig& config) {
        m_filterConfig = config;
        for (auto& dev : m_devices) {
            if (dev) dev->UpdateFilterConfig(m_filterConfig);
        }
    }

    void ToggleDisparity() {
        m_filterConfig.enableDisparity = !m_filterConfig.enableDisparity;
        std::cout << "[Depth Filter] Disparity Domain Filtering: " << (m_filterConfig.enableDisparity ? "ON" : "OFF") << std::endl;
        UpdateFilterConfig(m_filterConfig);
    }

    void ToggleSpatial() {
        m_filterConfig.enableSpatial = !m_filterConfig.enableSpatial;
        std::cout << "[Depth Filter] Spatial Filtering: " << (m_filterConfig.enableSpatial ? "ON" : "OFF") << std::endl;
        UpdateFilterConfig(m_filterConfig);
    }

    void ToggleTemporal() {
        m_filterConfig.enableTemporal = !m_filterConfig.enableTemporal;
        std::cout << "[Depth Filter] Temporal Filtering: " << (m_filterConfig.enableTemporal ? "ON" : "OFF") << std::endl;
        UpdateFilterConfig(m_filterConfig);
    }

    void ToggleHoleFilling() {
        m_filterConfig.enableHoleFilling = !m_filterConfig.enableHoleFilling;
        std::cout << "[Depth Filter] Hole Filling: " << (m_filterConfig.enableHoleFilling ? "ON" : "OFF") << std::endl;
        UpdateFilterConfig(m_filterConfig);
    }

    const RealSenseFilterConfig& GetFilterConfig() const {
        return m_filterConfig;
    }

    bool Start(std::vector<InputCamera>& inputCameras, const RealSenseFilterConfig& filterConfig) {
        m_filterConfig = filterConfig;
        m_devices.clear();

        rs2::context ctx;
        std::vector<std::string> availSerials;
        std::vector<std::string> availModels;

        for (int retry = 0; retry < 5; ++retry) {
            try {
                auto discovered = ctx.query_devices();
                availSerials.clear();
                availModels.clear();

                for (const auto& dev : discovered) {
                    std::string s = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
                    std::string m = dev.get_info(RS2_CAMERA_INFO_NAME);
                    availSerials.push_back(s);
                    availModels.push_back(m);
                }
                if (!availSerials.empty()) break;
            } catch (const std::exception& e) {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
        }

        std::cout << "\n===========================================================" << std::endl;
        std::cout << " [RealSense Hardware Discovery & Role Assignment]" << std::endl;
        std::cout << "===========================================================" << std::endl;
        std::cout << "Detected " << availSerials.size() << " RealSense physical device(s):" << std::endl;

        for (size_t d = 0; d < availSerials.size(); ++d) {
            std::cout << "  - " << availModels[d] << " | Serial: " << availSerials[d] << std::endl;
        }
        std::cout << "-----------------------------------------------------------" << std::endl;

        if (availSerials.empty()) {
            std::cerr << "[RealSense] Error: No Intel RealSense devices detected!" << std::endl;
            return false;
        }

        try {

            // Create devices for each rendering camera requested in inputCameras
            for (size_t i = 0; i < inputCameras.size(); ++i) {
                auto& cam = inputCameras[i];
                std::string targetSerial = cam.serial_number;
                std::string targetRole = cam.role.empty() ? ("render_" + std::to_string(i)) : cam.role;

                // Auto-match serial if not explicitly specified in JSON
                if (targetSerial.empty()) {
                    // Match next available D455 rendering device
                    for (size_t d = 0; d < availSerials.size(); ++d) {
                        bool alreadyUsed = false;
                        for (const auto& devPtr : m_devices) {
                            if (devPtr->GetSerialNumber() == availSerials[d]) {
                                alreadyUsed = true;
                                break;
                            }
                        }
                        if (!alreadyUsed && availModels[d].find("D455") != std::string::npos) {
                            targetSerial = availSerials[d];
                            break;
                        }
                    }
                    if (targetSerial.empty() && i < availSerials.size()) {
                        targetSerial = availSerials[i];
                    }
                }

                // Check if target serial is connected
                bool serialFound = false;
                for (const auto& s : availSerials) {
                    if (s == targetSerial) {
                        serialFound = true;
                        break;
                    }
                }
                if (!serialFound && !targetSerial.empty()) {
                    std::cerr << "\n[RealSense Connection Warning] Configured camera [" << i << "] (Role: '" << targetRole 
                              << "', Serial: " << targetSerial << ") is NOT connected to any USB port!" << std::endl;
                    std::cerr << " -> Please check/re-plug the physical USB cable for camera '" << targetRole 
                              << "' (S/N " << targetSerial << ")." << std::endl;
                }

                auto device = std::unique_ptr<RealSenseDevice>(new RealSenseDevice(targetSerial, targetRole));
                std::cout << "[Role Assignment] Camera [" << i << "]: Role='" << targetRole 
                          << "' -> Serial: " << targetSerial << std::endl;

                int w = (cam.res_x > 0) ? cam.res_x : 640;
                int h = (cam.res_y > 0) ? cam.res_y : 480;
                float zn = (cam.z_near > 0.0f) ? cam.z_near : 0.2f;
                float zf = (cam.z_far > zn) ? cam.z_far : 5.0f;

                if (!device->Start(w, h, 30, zn, zf, m_filterConfig)) {
                    std::cerr << "[RealSense] Failed to start device for camera [" << i << "] (Serial: " << targetSerial << ")" << std::endl;
                    return false;
                }

                // Update InputCamera intrinsics with exact factory-calibrated color intrinsics
                const auto& intr = device->GetIntrinsics();
                cam.focal_x = intr.fx;
                cam.focal_y = intr.fy;
                cam.principal_point_x = intr.ppx;
                cam.principal_point_y = intr.ppy;
                cam.res_x = intr.width;
                cam.res_y = intr.height;

                std::cout << "  Intrinsics loaded: fx=" << std::fixed << std::setprecision(2) << intr.fx 
                          << ", fy=" << intr.fy << ", cx=" << intr.ppx << ", cy=" << intr.ppy << std::endl;
                std::cout << "  Extrinsics loaded: pos=[" << std::setprecision(4) << cam.pos.x << ", " 
                          << cam.pos.y << ", " << cam.pos.z << "] m | rot=[" << cam.rot.x << ", " << cam.rot.y << ", " << cam.rot.z << "] rad" << std::endl;

                m_devices.push_back(std::move(device));
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }

            if (m_filterConfig.enableRgbBridge) {
                m_rgbServer.Start(m_filterConfig.rgbBridgePort);
            }

            m_isStreaming = true;
            m_globalFrameCount = 0;

            std::cout << "===========================================================" << std::endl;
            std::cout << " [RealSense Multi-View RGB-D Pipeline ACTIVE (" << m_devices.size() << " Cameras)]" << std::endl;
            std::cout << "===========================================================\n" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[RealSense Manager] Exception: " << e.what() << std::endl;
            m_isStreaming = false;
            return false;
        }
    }

    // Single-camera Start overload for backward compatibility
    bool Start(int width = 640, int height = 480, int fps = 30, float zNear = 0.2f, float zFar = 5.0f) {
        std::vector<InputCamera> defaultCams;
        InputCamera c;
        c.res_x = width;
        c.res_y = height;
        c.z_near = zNear;
        c.z_far = zFar;
        c.role = "render_main";
        defaultCams.push_back(c);
        return Start(defaultCams, m_filterConfig);
    }

    void Stop() {
        m_rgbServer.Stop();
        for (auto& d : m_devices) {
            if (d) d->Stop();
        }
        m_devices.clear();
        m_isStreaming = false;
        std::cout << "[RealSense] All pipelines stopped." << std::endl;
    }

    // Capture frames from all active rendering RealSense cameras simultaneously
    bool CaptureAllFrames(std::vector<const uint8_t*>& outRgbList, std::vector<const uint16_t*>& outDepthList, unsigned int timeoutMs = 2000) {
        if (!m_isStreaming || m_devices.empty()) return false;

        outRgbList.resize(m_devices.size());
        outDepthList.resize(m_devices.size());

        bool allOk = true;
        bool mainOk = false;
        for (size_t i = 0; i < m_devices.size(); ++i) {
            const uint8_t* pRgb = nullptr;
            const uint16_t* pDepth = nullptr;
            if (m_devices[i]->CaptureNextFrame(pRgb, pDepth, timeoutMs)) {
                outRgbList[i] = pRgb;
                outDepthList[i] = pDepth;
                if (i == 0) mainOk = true;
            } else {
                allOk = false;
            }
        }   
        // Push to the eye tracker bridge as soon as the MAIN camera succeeds,
    // independent of whether a secondary camera also captured this cycle.
    // A dropped/slow secondary frame should not starve the eye tracker,
    // which only ever needs main's RGB-D.
        if (mainOk && !outRgbList.empty() && outRgbList[0] != nullptr) {
                m_globalFrameCount++;
                if (m_filterConfig.enableRgbBridge) {
                    size_t rgbBytes = (size_t)m_devices[0]->GetWidth() * m_devices[0]->GetHeight() * 3;
                    size_t depthBytes = (size_t)m_devices[0]->GetWidth() * m_devices[0]->GetHeight() * sizeof(uint16_t);
                    m_rgbServer.PushRgbdFrame(outRgbList[0], m_devices[0]->GetRawDepth(), rgbBytes, depthBytes);
                }
        }

    

        return allOk;
    }

    // Single-frame capture for fallback
    bool CaptureNextFrame(const uint8_t*& outRgb, const uint16_t*& outNormalizedDepth, unsigned int timeoutMs = 2000) {
        std::vector<const uint8_t*> rgbs;
        std::vector<const uint16_t*> depths;
        if (CaptureAllFrames(rgbs, depths, timeoutMs) && !rgbs.empty()) {
            outRgb = rgbs[0];
            outNormalizedDepth = depths[0];
            return true;
        }
        return false;
    }

    size_t GetDeviceCount() const { return m_devices.size(); }
    bool IsStreaming() const { return m_isStreaming; }
    uint64_t GetFrameCount() const { return m_globalFrameCount; }

    const RealSenseDevice* GetDevice(size_t index) const {
        if (index < m_devices.size()) return m_devices[index].get();
        return nullptr;
    }
};

#endif // REALSENSE_RECEIVER_H
