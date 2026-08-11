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

class EyeTrackerReceiver {
private:
    SOCKET m_socket = INVALID_SOCKET;
    bool m_initialized = false;
    bool m_hasFirstPacket = false;
    
    int m_latestState = 2; // default LOST
    std::chrono::steady_clock::time_point m_lastPacketTime = std::chrono::steady_clock::now();

    // Calibration reference point (in mm)
    float m_refX = 0.0f;
    float m_refY = 0.0f;
    float m_refZ = 0.0f;

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
        std::cout << "[EyeTracker] Listening for UDP packets on port " << port << "..." << std::endl;
        return true;
    }

    // Read all pending packets to get the absolute latest tracking position
    bool GetLatestEyeOffset(float& outDx, float& outDy, float& outDz, float scaleX = 1.0f, float scaleY = 1.0f, float scaleZ = 1.0f) {
        if (!m_initialized) return false;

        char buffer[256];
        bool gotNewPacket = false;
        float rx = 0.0f, ry = 0.0f, rz = 0.0f;

        while (true) {
            int bytesReceived = recv(m_socket, buffer, sizeof(buffer) - 1, 0);
            if (bytesReceived > 0) {
                buffer[bytesReceived] = '\0';
                float tx, ty, tz;
                int stateVal = 0;
                int parsed = sscanf(buffer, "%f %f %f %d", &tx, &ty, &tz, &stateVal);
                if (parsed >= 3) {
                    rx = tx;
                    ry = ty;
                    rz = tz;
                    if (parsed == 4) {
                        m_latestState = stateVal;
                    } else {
                        m_latestState = 0; // default TRACKING
                    }
                    gotNewPacket = true;
                }
            } else {
                break;
            }
        }

        if (gotNewPacket) {
            m_lastPacketTime = std::chrono::steady_clock::now();
            if (!m_hasFirstPacket) {
                m_refX = rx;
                m_refY = ry;
                m_refZ = rz;
                m_hasFirstPacket = true;
                m_latestState = 0; // successfully calibrated
                std::cout << "[EyeTracker] Calibration reference established at (" 
                          << m_refX << ", " << m_refY << ", " << m_refZ << ") mm." << std::endl;
            }

            // Convert delta to meters (divide by 1000.0)
            // Negate Y because in OpenGL +Y is up, but MediaPipe coordinates increase downwards
            outDx = ((rx - m_refX) / 1000.0f) * scaleX;
            outDy = (-(ry - m_refY) / 1000.0f) * scaleY;
            outDz = ((rz - m_refZ) / 1000.0f) * scaleZ;
            return true;
        } else {
            // Check packet timeout (500 ms)
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastPacketTime).count();
            if (duration > 500) {
                m_latestState = 2; // LOST
            }
        }

        return false;
    }

    int GetLatestState() const {
        return m_latestState;
    }

    float GetBaselineDistance() const {
        if (!m_hasFirstPacket) return 0.6f;
        return m_refZ / 1000.0f; // convert mm to meters
    }

    void ResetCalibration() {
        m_hasFirstPacket = false;
        m_latestState = 3; // RECALIBRATING
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
            m_hasFirstPacket = false;
        }
    }
};

#endif // EYE_TRACKER_RECEIVER_H
