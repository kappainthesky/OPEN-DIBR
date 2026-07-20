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

class EyeTrackerReceiver {
private:
    SOCKET m_socket = INVALID_SOCKET;
    bool m_initialized = false;
    bool m_hasFirstPacket = false;
    
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
                if (sscanf(buffer, "%f %f %f", &tx, &ty, &tz) == 3) {
                    rx = tx;
                    ry = ty;
                    rz = tz;
                    gotNewPacket = true;
                }
            } else {
                break;
            }
        }

        if (gotNewPacket) {
            if (!m_hasFirstPacket) {
                m_refX = rx;
                m_refY = ry;
                m_refZ = rz;
                m_hasFirstPacket = true;
                std::cout << "[EyeTracker] Calibration reference established at (" 
                          << m_refX << ", " << m_refY << ", " << m_refZ << ") mm." << std::endl;
            }

            // Convert delta to meters (divide by 1000.0)
            // Negate Y because in OpenGL +Y is up, but MediaPipe coordinates increase downwards
            outDx = ((rx - m_refX) / 1000.0f) * scaleX;
            outDy = (-(ry - m_refY) / 1000.0f) * scaleY;
            outDz = ((rz - m_refZ) / 1000.0f) * scaleZ;
            return true;
        }

        return false;
    }

    void ResetCalibration() {
        m_hasFirstPacket = false;
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
