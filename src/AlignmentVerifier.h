#ifndef ALIGNMENT_VERIFIER_H
#define ALIGNMENT_VERIFIER_H

#include <GL/glew.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>

class AlignmentVerifier {
public:
    static void Verify(GLuint texColor, GLuint texDepth, int width, int height, int bitdepth, float z_near, float z_far, int frameNr) {
        // Run once on initial frame (frame 0) to verify alignment without runtime overhead
        if (frameNr != 0) return;

        int pixelCount = width * height;
        std::vector<float> luma(pixelCount, 0.0f);
        std::vector<float> depthMeters(pixelCount, 0.0f);

        // 1. Download Luma (Y plane stored in GL_RED of NV12 texture)
        glBindTexture(GL_TEXTURE_2D, texColor);
        int roundedHeight = ((height + 16 - 1) / 16) * 16;
        int colorBufferSize = width * (roundedHeight + height / 2);
        std::vector<uint8_t> colorBuffer(colorBufferSize);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, colorBuffer.data());
        for (int i = 0; i < pixelCount; ++i) {
            luma[i] = (float)colorBuffer[i] / 255.0f;
        }

        // 2. Download and deproject Depth to physical meters
        glBindTexture(GL_TEXTURE_2D, texDepth);
        uint16_t minRaw = 65535, maxRaw = 0;
        double sumRaw = 0.0;
        int nonZeroCount = 0;
        if (bitdepth > 8) {
            std::vector<uint16_t> depthBuffer(pixelCount);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_SHORT, depthBuffer.data());
            for (int i = 0; i < pixelCount; ++i) {
                uint16_t val = depthBuffer[i];
                if (val > 0) {
                    if (val < minRaw) minRaw = val;
                    if (val > maxRaw) maxRaw = val;
                    sumRaw += val;
                    nonZeroCount++;
                }
                float normVal = (float)val / 65535.0f;
                if (normVal > 0.0f) {
                    depthMeters[i] = 1.0f / (1.0f / z_far + normVal * (1.0f / z_near - 1.0f / z_far));
                }
            }
        } else {
            std::vector<uint8_t> depthBuffer(pixelCount);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, depthBuffer.data());
            for (int i = 0; i < pixelCount; ++i) {
                uint8_t val = depthBuffer[i];
                if (val > 0) {
                    if (val < minRaw) minRaw = val;
                    if (val > maxRaw) maxRaw = val;
                    sumRaw += val;
                    nonZeroCount++;
                }
                float normVal = (float)val / 255.0f;
                if (normVal > 0.0f) {
                    depthMeters[i] = 1.0f / (1.0f / z_far + normVal * (1.0f / z_near - 1.0f / z_far));
                }
            }
        }
        std::cout << "[AlignmentVerifier] Depth Stats -> Raw Min: " << minRaw << ", Max: " << maxRaw 
                  << ", Avg: " << (nonZeroCount > 0 ? sumRaw / nonZeroCount : 0.0)
                  << ", NonZero Pixels: " << nonZeroCount << "/" << pixelCount << std::endl;

        // 3. Sobel Edge Extraction
        std::vector<bool> rgbEdges(pixelCount, false);
        std::vector<int> depthEdgeIndices;

        float t_rgb = 0.08f;
        float t_depth = 0.15f; // 15 cm discontinuity threshold

        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;

                // Color gradient
                float dx_rgb = luma[idx + 1] - luma[idx - 1];
                float dy_rgb = luma[idx + width] - luma[idx - width];
                float grad_rgb = std::sqrt(dx_rgb * dx_rgb + dy_rgb * dy_rgb);
                if (grad_rgb > t_rgb) {
                    rgbEdges[idx] = true;
                }

                // Depth gradient
                float z_curr = depthMeters[idx];
                if (z_curr > 0.0f && z_curr < 999.0f) {
                    float z_r = depthMeters[idx + 1];
                    float z_l = depthMeters[idx - 1];
                    float z_b = depthMeters[idx + width];
                    float z_t = depthMeters[idx - width];

                    if (z_r > 0.0f && z_l > 0.0f && z_b > 0.0f && z_t > 0.0f) {
                        float dx_depth = std::abs(z_r - z_l);
                        float dy_depth = std::abs(z_b - z_t);
                        float grad_depth = std::max(dx_depth, dy_depth);
                        if (grad_depth > t_depth) {
                            depthEdgeIndices.push_back(idx);
                        }
                    }
                }
            }
        }

        if (depthEdgeIndices.empty()) return;

        // 4. Chamfer distance check for co-occurring edges
        auto calcScoreForShift = [&](int dx, int dy) -> float {
            int alignedCount = 0;
            for (int idx : depthEdgeIndices) {
                int ey = idx / width;
                int ex = idx % width;

                int sx = ex + dx;
                int sy = ey + dy;

                if (sx < 2 || sx >= width - 2 || sy < 2 || sy >= height - 2) continue;

                // Search 3x3 window around target shifted position
                bool found = false;
                for (int wy = -1; wy <= 1; ++wy) {
                    for (int wx = -1; wx <= 1; ++wx) {
                        int nidx = (sy + wy) * width + (sx + wx);
                        if (rgbEdges[nidx]) {
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                if (found) alignedCount++;
            }
            return (float)alignedCount / depthEdgeIndices.size();
        };

        float baseScore = calcScoreForShift(0, 0);
        GetLastScoreRef() = baseScore * 100.0f;

        // 5. Shift Diagnosis: Search translation space for offset maximums
        int bestDx = 0;
        int bestDy = 0;
        float bestScore = baseScore;

        std::vector<int> testShifts = {-8, -6, -4, -2, 2, 4, 6, 8};
        for (int dy : testShifts) {
            for (int dx : testShifts) {
                float score = calcScoreForShift(dx, dy);
                if (score > bestScore) {
                    bestScore = score;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }

        // 6. Diagnostics Reporting
        std::cout << "[AlignmentVerifier] Frame " << frameNr << " | Alignment Score: " 
                  << std::fixed << std::setprecision(1) << (baseScore * 100.0f) << "%" << std::endl;

        if (bestScore > baseScore + 0.12f && (bestDx != 0 || bestDy != 0)) {
            std::cout << "[WARNING] Spatial misalignment detected between RGB and Depth!" << std::endl;
            std::cout << " -> Diagnosed Shift: The depth map is shifted by (" << bestDx << ", " << bestDy << ") pixels relative to color map." << std::endl;
            std::cout << " -> Alignment Score would improve from " << (baseScore * 100.0f) << "% to " << (bestScore * 100.0f) << "% if shifted." << std::endl;
            std::cout << " -> Artifact: Color bleeding, ghost silhouettes, and jagged warping artifacts at foreground boundaries." << std::endl;
        } else if (baseScore < 0.50f) {
            std::cout << "[WARNING] Poor RGB-Depth alignment quality (Score: " << (baseScore * 100.0f) << "%)." << std::endl;
            std::cout << " -> Artifact: Unsynchronized depth boundaries will cause tearing and stretched triangles to bleed into background regions." << std::endl;
        } else {
            std::cout << "[INFO] RGB-Depth alignment quality is good (Score: " << (baseScore * 100.0f) << "%)." << std::endl;
        }
    }

    static float GetLastScore() {
        return GetLastScoreRef();
    }

private:
    static float& GetLastScoreRef() {
        static float score = 0.0f;
        return score;
    }
};

#endif // ALIGNMENT_VERIFIER_H
