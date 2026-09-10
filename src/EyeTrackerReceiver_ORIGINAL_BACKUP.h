#ifndef EYE_TRACKER_RECEIVER_H
#define EYE_TRACKER_RECEIVER_H

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
typedef int SOCKET;
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#endif

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <algorithm>

class EyeTrackerReceiver {
private:
    SOCKET m_socket = INVALID_SOCKET;
    bool m_initialized = false;
    bool m_hasReceivedPacket = false;
    
    int m_latestState = 2; // default LOST (0=TRACKING, 1=LOW_CONFIDENCE, 2=LOST)
    std::chrono::steady_clock::time_point m_lastPacketTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point m_lastUpdateTime = std::chrono::steady_clock::now();

    // Baseline viewing distance & center from screen in millimeters
    float m_baselineX = 0.0f;
    float m_baselineY = 0.0f;
    float m_baselineDistanceMm = 650.0f;

    // Filtered output position (in meters for OpenDIBR camera offset)
    float m_curDx = 0.0f;
    float m_curDy = 0.0f;
    float m_curDz = 0.0f;

    // Previous target position for velocity estimation
    float m_prevTargetDx = 0.0f;
    float m_prevTargetDy = 0.0f;

    // Latest raw screen-space coordinates received from tracker (in mm)
    float m_latestRawX = 0.0f;
    float m_latestRawY = 0.0f;
    float m_latestRawZ = 650.0f;

    float m_latestLeftX = 0.0f, m_latestLeftY = 0.0f, m_latestLeftZ = 0.0f;
    bool m_leftTracking = false;

    float m_latestRightX = 0.0f, m_latestRightY = 0.0f, m_latestRightZ = 0.0f;
    bool m_rightTracking = false;

    // Telemetry from tracker
    float m_latestCaptureTime = 0.0f;
    float m_latestMpTimeMs = 0.0f;
    float m_latestTrackFps = 30.0f;
    float m_latestLatencyMs = 0.0f;

    // Tuning & Responsiveness Parameters
    float m_deadzoneX = 1.5f;          // 1.5mm deadzone for stationary micro-jitter rejection
    float m_deadzoneY = 1.5f;          // 1.5mm deadzone
    float m_deadzoneZ = 4.0f;          // 4mm deadzone
    float m_smoothAlphaStill = 0.40f;  // Alpha when still (smooth, steady resting)
    float m_smoothAlphaMove = 0.95f;   // Alpha when moving (instantaneous 1:1 VR-like response)
    float m_trackingGain = 1.0f;       // 1.0x gain for full-scale physical head-tracking parity
    float m_maxClampMeters = 0.15f;    // Generous +-0.15 m (+-15 cm) clamp

    // Motion Prediction (disabled temporarily for stability)
    bool m_enablePrediction = false;
    float m_predTimeSec = 0.025f;      // 25 ms prediction

    // Interactive axis & debug toggles
    bool m_enableViewpointUpdate = true;
    bool m_enableX = true;
    bool m_enableY = true;
    bool m_enableZ = false;            // Z tracking explicitly disabled
    bool m_invertX = false;
    bool m_invertY = false;

public:
    EyeTrackerReceiver() {}
    ~EyeTrackerReceiver() { Cleanup(); }

    bool Init(int port = 9999) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "[EyeTracker] WSAStartup failed." << std::endl;
            return false;
        }
#endif

        m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (m_socket == INVALID_SOCKET) {
            std::cerr << "[EyeTracker] Socket creation failed." << std::endl;
#ifdef _WIN32
            WSACleanup();
#endif
            return false;
        }

        // Set non-blocking mode
#ifdef _WIN32
        u_long mode = 1;
        if (ioctlsocket(m_socket, FIONBIO, &mode) != 0) {
            std::cerr << "[EyeTracker] Failed to set non-blocking mode." << std::endl;
            closesocket(m_socket);
            WSACleanup();
            return false;
        }
#else
        int flags = fcntl(m_socket, F_GETFL, 0);
        if (flags == -1 || fcntl(m_socket, F_SETFL, flags | O_NONBLOCK) != 0) {
            std::cerr << "[EyeTracker] Failed to set non-blocking mode." << std::endl;
            close(m_socket);
            return false;
        }
#endif

        int optval = 1;
        setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, (const char*)&optval, sizeof(optval));
#ifdef SO_REUSEPORT
        setsockopt(m_socket, SOL_SOCKET, SO_REUSEPORT, (const char*)&optval, sizeof(optval));
#endif

        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);

        if (bind(m_socket, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
            std::cerr << "[EyeTracker] Bind failed on port " << port << "." << std::endl;
#ifdef _WIN32
            closesocket(m_socket);
            WSACleanup();
#else
            close(m_socket);
#endif
            return false;
        }

        m_initialized = true;
        std::cout << "[EyeTracker] Low-latency UDP receiver initialized on port " << port 
                  << " (Deadzone: 3mm, Alpha: 0.50-0.85, Prediction: 25ms)." << std::endl;
        return true;
    }

    // Read and drain all pending UDP packets non-blocking to get the absolute newest tracking position
    bool GetLatestEyeOffset(float& outDx, float& outDy, float& outDz, float scaleX = 1.0f, float scaleY = 1.0f, float scaleZ = 1.0f) {
        if (!m_initialized) return false;

        char buffer[512];
        bool gotNewPacket = false;
        float rx = 0.0f, ry = 0.0f, rz = 0.0f;
        float t_cap = 0.0f, t_mp = 0.0f, trk_fps = 30.0f;

        // Drain loop: keep only the newest packet
        while (true) {
            int bytesReceived = recv(m_socket, buffer, sizeof(buffer) - 1, 0);
            if (bytesReceived > 0) {
                buffer[bytesReceived] = '\0';
                float tx = 0, ty = 0, tz = 0;
                int stateVal = 0;
                float lx = 0, ly = 0, lz = 0, rx_pos = 0, ry_pos = 0, rz_pos = 0;
                int l_state = 0, r_state = 0;
                float packet_t_cap = 0.0f, packet_t_mp = 0.0f, packet_fps = 0.0f;

                int parsed = sscanf(buffer, "%f %f %f %d %f %f %f %d %f %f %f %d %f %f %f", 
                                    &tx, &ty, &tz, &stateVal,
                                    &lx, &ly, &lz, &l_state,
                                    &rx_pos, &ry_pos, &rz_pos, &r_state,
                                    &packet_t_cap, &packet_t_mp, &packet_fps);

                if (parsed >= 3) {
                    if (std::isfinite(tx) && std::isfinite(ty) && std::isfinite(tz)) {
                        rx = tx;
                        ry = ty;
                        rz = tz;
                        m_latestRawX = rx;
                        m_latestRawY = ry;
                        m_latestRawZ = rz;
                        m_latestState = (parsed >= 4) ? stateVal : 0;

                        if (parsed >= 12) {
                            m_latestLeftX = lx;
                            m_latestLeftY = ly;
                            m_latestLeftZ = lz;
                            m_leftTracking = (l_state == 0);

                            m_latestRightX = rx_pos;
                            m_latestRightY = ry_pos;
                            m_latestRightZ = rz_pos;
                            m_rightTracking = (r_state == 0);
                        }

                        if (parsed >= 15) {
                            t_cap = packet_t_cap;
                            t_mp = packet_t_mp;
                            trk_fps = packet_fps;
                            m_latestCaptureTime = t_cap;
                            m_latestMpTimeMs = t_mp;
                            m_latestTrackFps = trk_fps;
                        }
                        gotNewPacket = true;
                    }
                }
            } else {
                break;
            }
        }

        auto now = std::chrono::steady_clock::now();

        if (gotNewPacket) {
            m_lastPacketTime = now;
            if (!m_hasReceivedPacket) {
                m_hasReceivedPacket = true;
                m_baselineX = rx;
                m_baselineY = ry;
                if (rz > 200.0f && rz < 3000.0f) {
                    m_baselineDistanceMm = rz;
                }
                std::cout << "[EyeTracker] Tracking packet locked in (baseline center: X=" << m_baselineX << ", Y=" << m_baselineY 
                          << ", Z=" << m_baselineDistanceMm << " mm)." << std::endl;
            }

            // 1. Delta offset relative to calibrated baseline
            float deltaX_mm = rx - m_baselineX;
            float deltaY_mm = ry - m_baselineY;

            // Deadzone around baseline in mm (3mm deadzone)
            if (std::abs(deltaX_mm) < m_deadzoneX) deltaX_mm = 0.0f;
            if (std::abs(deltaY_mm) < m_deadzoneY) deltaY_mm = 0.0f;

            // 2. Convert from millimeters to meters with scaling and gain
            float xMeters = (deltaX_mm * 0.001f) * m_trackingGain * scaleX;
            float yMeters = (deltaY_mm * 0.001f) * m_trackingGain * scaleY;

            // 3. Axis routing & direction conventions
            float targetDx = 0.0f;
            float targetDy = 0.0f;
            float targetDz = 0.0f;

            if (m_enableViewpointUpdate) {
                if (m_enableX) {
                    targetDx = m_invertX ? -xMeters : xMeters;
                }
                if (m_enableY) {
                    targetDy = m_invertY ? yMeters : -yMeters;
                }
                if (m_enableZ) {
                    targetDz = (rz - m_baselineDistanceMm) * 0.001f * m_trackingGain * scaleZ;
                }
            }

            // 4. Safety Clamping
            targetDx = std::max(-m_maxClampMeters, std::min(m_maxClampMeters, targetDx));
            targetDy = std::max(-m_maxClampMeters, std::min(m_maxClampMeters, targetDy));
            targetDz = std::max(-m_maxClampMeters, std::min(m_maxClampMeters, targetDz));

            // 5. Dynamic Velocity & Short-Term Extrapolation
            float dt = std::max(0.001f, std::chrono::duration<float>(now - m_lastUpdateTime).count());
            m_lastUpdateTime = now;

            float vx = (targetDx - m_prevTargetDx) / dt;
            float vy = (targetDy - m_prevTargetDy) / dt;
            m_prevTargetDx = targetDx;
            m_prevTargetDy = targetDy;

            float speed = std::sqrt(vx * vx + vy * vy); // m/s

            // Adaptive alpha: stable when still (0.40), ultra-responsive when moving (0.95)
            float alpha_eff = m_smoothAlphaStill + (m_smoothAlphaMove - m_smoothAlphaStill) * std::min(1.0f, speed / 0.015f);

            float predDx = targetDx;
            float predDy = targetDy;
            if (m_enablePrediction && speed > 0.015f) {
                predDx += vx * m_predTimeSec;
                predDy += vy * m_predTimeSec;
                predDx = std::max(-m_maxClampMeters, std::min(m_maxClampMeters, predDx));
                predDy = std::max(-m_maxClampMeters, std::min(m_maxClampMeters, predDy));
            }

            // 6. Responsive EMA Smoothing
            m_curDx = alpha_eff * predDx + (1.0f - alpha_eff) * m_curDx;
            m_curDy = alpha_eff * predDy + (1.0f - alpha_eff) * m_curDy;
            m_curDz = 0.0f; // Z disabled

            outDx = m_curDx;
            outDy = m_curDy;
            outDz = m_curDz;
            return true;
        } else {
            // Check packet timeout (hold position for first 500 ms)
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastPacketTime).count();
            if (duration > 500) {
                m_latestState = 2; // LOST
                // Only slowly decay back to zero offset after prolonged loss (>500ms)
                m_curDx = 0.98f * m_curDx;
                m_curDy = 0.98f * m_curDy;
                m_curDz = 0.98f * m_curDz;
            }

            outDx = m_curDx;
            outDy = m_curDy;
            outDz = m_curDz;
        }

        return false;
    }

    bool GetLatestRawPosition(float& rx, float& ry, float& rz) const {
        rx = m_latestRawX;
        ry = m_latestRawY;
        rz = m_latestRawZ;
        return m_hasReceivedPacket;
    }

    bool GetDetailedTelemetry(float& rawX, float& rawY, float& rawZ,
                              float& convX, float& convY, float& convZ,
                              bool& vpEnabled, bool& xEnabled, bool& yEnabled, bool& zEnabled,
                              bool& xInv, bool& yInv, bool& predEnabled,
                              float& trackFps, float& mpTimeMs) const {
        rawX = m_latestRawX;
        rawY = m_latestRawY;
        rawZ = m_latestRawZ;
        convX = m_curDx;
        convY = m_curDy;
        convZ = m_curDz;
        vpEnabled = m_enableViewpointUpdate;
        xEnabled = m_enableX;
        yEnabled = m_enableY;
        zEnabled = m_enableZ;
        xInv = m_invertX;
        yInv = m_invertY;
        predEnabled = m_enablePrediction;
        trackFps = m_latestTrackFps;
        mpTimeMs = m_latestMpTimeMs;
        return m_hasReceivedPacket;
    }

    int GetLatestState() const {
        return m_latestState;
    }

    float GetBaselineDistance() const {
        return m_baselineDistanceMm / 1000.0f; // in meters
    }

    // Interactive Debug Hotkey Controls
    void TogglePrediction() {
        m_enablePrediction = !m_enablePrediction;
        std::cout << "[EyeTracker Debug] 25ms Velocity Prediction " << (m_enablePrediction ? "ENABLED" : "DISABLED") << std::endl;
    }

    void ToggleViewpointUpdate() {
        m_enableViewpointUpdate = !m_enableViewpointUpdate;
        std::cout << "[EyeTracker Debug] Viewpoint Update " << (m_enableViewpointUpdate ? "ENABLED" : "DISABLED") << std::endl;
    }

    void ToggleXAxis() {
        m_enableX = !m_enableX;
        std::cout << "[EyeTracker Debug] X-Axis Tracking " << (m_enableX ? "ENABLED" : "DISABLED") << std::endl;
    }

    void ToggleYAxis() {
        m_enableY = !m_enableY;
        std::cout << "[EyeTracker Debug] Y-Axis Tracking " << (m_enableY ? "ENABLED" : "DISABLED") << std::endl;
    }

    void ToggleInvertX() {
        m_invertX = !m_invertX;
        std::cout << "[EyeTracker Debug] X-Axis Direction: " << (m_invertX ? "INVERTED (-X)" : "NORMAL (+X)") << std::endl;
    }

    void ToggleInvertY() {
        m_invertY = !m_invertY;
        std::cout << "[EyeTracker Debug] Y-Axis Direction: " << (m_invertY ? "INVERTED (+Y)" : "NORMAL (-Y)") << std::endl;
    }

    void IncreaseGain() {
        m_trackingGain = std::min(3.0f, m_trackingGain + 0.15f);
        std::cout << "[EyeTracker] Tracking Gain increased to: " << std::fixed << std::setprecision(2) << m_trackingGain << "x" << std::endl;
    }

    void DecreaseGain() {
        m_trackingGain = std::max(0.1f, m_trackingGain - 0.15f);
        std::cout << "[EyeTracker] Tracking Gain decreased to: " << std::fixed << std::setprecision(2) << m_trackingGain << "x" << std::endl;
    }

    void IncreaseClamp() {
        m_maxClampMeters = std::min(0.30f, m_maxClampMeters + 0.02f);
        std::cout << "[EyeTracker] Max Viewpoint Clamp increased to: " << std::fixed << std::setprecision(2) << m_maxClampMeters << " m (" 
                  << (m_maxClampMeters * 100.0f) << " cm)" << std::endl;
    }

    void DecreaseClamp() {
        m_maxClampMeters = std::max(0.02f, m_maxClampMeters - 0.02f);
        std::cout << "[EyeTracker] Max Viewpoint Clamp decreased to: " << std::fixed << std::setprecision(2) << m_maxClampMeters << " m (" 
                  << (m_maxClampMeters * 100.0f) << " cm)" << std::endl;
    }

    float GetGain() const { return m_trackingGain; }
    float GetClamp() const { return m_maxClampMeters; }

    void ResetCalibration() {
        m_baselineX = m_latestRawX;
        m_baselineY = m_latestRawY;
        if (m_latestRawZ > 200.0f && m_latestRawZ < 3000.0f) {
            m_baselineDistanceMm = m_latestRawZ;
        }
        m_curDx = 0.0f;
        m_curDy = 0.0f;
        m_curDz = 0.0f;
        m_prevTargetDx = 0.0f;
        m_prevTargetDy = 0.0f;
        std::cout << "[EyeTracker] Viewpoint offset reset to centered baseline (X=" << m_baselineX 
                  << ", Y=" << m_baselineY << ", Z=" << m_baselineDistanceMm << " mm)." << std::endl;
    }

    void Cleanup() {
        if (m_initialized) {
            if (m_socket != INVALID_SOCKET) {
#ifdef _WIN32
                closesocket(m_socket);
#else
                close(m_socket);
#endif
                m_socket = INVALID_SOCKET;
            }
#ifdef _WIN32
            WSACleanup();
#endif
            m_initialized = false;
            m_hasReceivedPacket = false;
        }
    }
};

#endif // EYE_TRACKER_RECEIVER_H
