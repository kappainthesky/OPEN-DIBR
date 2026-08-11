#ifndef PERFORMANCE_LOGGER_H
#define PERFORMANCE_LOGGER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <cuda.h>

struct FrameMetrics {
    int frame;
    double timestampSec;
    float fps;
    float frameTimeMs;
    float decodeTimeMs;
    float warpingTimeMs;
    float blendingTimeMs;
    float gpuMemoryMB;
    float triangleMargin;
    float eyeLatencyMs;
};

class PerformanceLogger {
private:
    static std::vector<FrameMetrics>& GetLogs() {
        static std::vector<FrameMetrics> logs;
        return logs;
    }

    static std::chrono::steady_clock::time_point& GetStartTime() {
        static std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        return start;
    }

public:
    static void LogFrame(int frame, float frameTimeMs, float fps,
                         float decodeTimeMs, float warpingTimeMs, float blendingTimeMs,
                         float triangleMargin, float eyeLatencyMs = 45.0f)
    {
        auto now = std::chrono::steady_clock::now();
        double timestampSec = std::chrono::duration<double>(now - GetStartTime()).count();

        // Query GPU Memory via CUDA Driver API
        size_t freeMem = 0, totalMem = 0;
        float gpuMemoryMB = 0.0f;
        if (cuMemGetInfo(&freeMem, &totalMem) == CUDA_SUCCESS) {
            gpuMemoryMB = (float)(totalMem - freeMem) / (1024.0f * 1024.0f);
        }

        static bool headerWritten = false;
        std::ofstream csv("performance_log.csv", std::ios::app);
        if (!headerWritten) {
            csv << "frame,timestamp_sec,fps,frame_time_ms,decode_time_ms,warping_time_ms,blending_time_ms,gpu_memory_mb,triangle_margin,eye_latency_ms\n";
            headerWritten = true;
        }

        csv << frame << ","
            << std::fixed << std::setprecision(4) << timestampSec << ","
            << std::setprecision(2) << fps << ","
            << frameTimeMs << ","
            << decodeTimeMs << ","
            << warpingTimeMs << ","
            << blendingTimeMs << ","
            << std::setprecision(1) << gpuMemoryMB << ","
            << triangleMargin << ","
            << eyeLatencyMs << "\n";

        csv.flush();
    }

    static void SaveToCSV(const std::string& filename = "performance_log.csv") {
        // Handled dynamically per frame in LogFrame
    }
};

#endif // PERFORMANCE_LOGGER_H
