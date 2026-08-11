#ifndef DATASET_VALIDATOR_H
#define DATASET_VALIDATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <memory>
#include <iomanip>
#include "ioHelper.h"
#include "FFmpegDemuxer.h"

// Forward declaration of Options to avoid circular dependencies
class Options;

class DatasetValidator {
public:
    static bool Validate(const std::vector<InputCamera>& inputCameras, bool usePNGs, bool useGStreamerInput = false, float default_z_near = 0.1f) {
        std::cout << "\n=============================================" << std::endl;
        std::cout << "        [DatasetValidator] Starting Sanity Check..." << std::endl;
        std::cout << "=============================================" << std::endl;

        bool hasCriticalError = false;
        int cameraCount = (int)inputCameras.size();

        // 1. Camera Count Check
        if (cameraCount == 0) {
            std::cout << "[CRITICAL ERROR] No input cameras loaded from the dataset JSON file!" << std::endl;
            std::cout << " -> Artifact: The rendering engine cannot start without at least one reference view." << std::endl;
            hasCriticalError = true;
            return false;
        }
        std::cout << "[INFO] Found " << cameraCount << " input camera configuration(s) in JSON." << std::endl;

        // Establish reference resolutions to check consistency across cameras
        int ref_res_x = inputCameras[0].res_x;
        int ref_res_y = inputCameras[0].res_y;

        for (int i = 0; i < cameraCount; i++) {
            const auto& cam = inputCameras[i];
            std::cout << "\n--- Validating Camera [" << i << "]: " << cam.pathColor << " ---" << std::endl;

            // 2. JSON Parameter Checks
            if (cam.res_x <= 0 || cam.res_y <= 0) {
                std::cout << "[CRITICAL ERROR] Invalid camera resolution in JSON: " << cam.res_x << "x" << cam.res_y << std::endl;
                std::cout << " -> Artifact: Division by zero or heap corruption in texture allocation." << std::endl;
                hasCriticalError = true;
            }

            if (cam.res_x != ref_res_x || cam.res_y != ref_res_y) {
                std::cout << "[WARNING] Resolution mismatch in JSON between Camera " << i << " (" << cam.res_x << "x" << cam.res_y 
                          << ") and Camera 0 (" << ref_res_x << "x" << ref_res_y << ")." << std::endl;
                std::cout << " -> Artifact: Mixing inputs of different resolutions can cause texture scaling mismatches or boundary stretching in DIBR blending." << std::endl;
            }

            // 3. Depth Range Validation
            if (cam.z_near <= 0.0f) {
                std::cout << "[WARNING] Invalid z_near depth range: " << cam.z_near << "m (must be > 0.0m)." << std::endl;
                std::cout << " -> Artifact: Projection math (division by zero) will fail, resulting in infinite depth stretching." << std::endl;
            }
            if (cam.z_far <= cam.z_near) {
                std::cout << "[WARNING] Invalid depth boundary: z_far (" << cam.z_far << ") is <= z_near (" << cam.z_near << ")." << std::endl;
                std::cout << " -> Artifact: Depth buffer sorting will be inverted, causing background pixels to render in front of foreground." << std::endl;
            }
            if (cam.z_near > 10.0f) {
                std::cout << "[WARNING] Unusually large z_near value: " << cam.z_near << "m." << std::endl;
                std::cout << " -> Artifact: Flat depth scaling might squash foreground features, reducing the 3D parallax effect." << std::endl;
            }

            // 4. File Existence Checks
            bool colorExists = FileExists(cam.pathColor);
            bool depthExists = FileExists(cam.pathDepth);

            if (!useGStreamerInput) {
                if (!colorExists) {
                    std::cout << "[CRITICAL ERROR] Missing color video/image file: " << cam.pathColor << std::endl;
                    std::cout << " -> Artifact: Rendering crash due to null pointer dereference in video demuxer/image loader." << std::endl;
                    hasCriticalError = true;
                }
                if (!depthExists) {
                    std::cout << "[CRITICAL ERROR] Missing depth video/image file: " << cam.pathDepth << std::endl;
                    std::cout << " -> Artifact: Missing geometry reference; 3D warping cannot be performed." << std::endl;
                    hasCriticalError = true;
                }
            } else {
                std::cout << "[INFO] Live GStreamer input active: bypassing disk file existence check for " << cam.pathColor << std::endl;
            }

            // 5. Video Metadata Checks (Only for dynamic video streams)
            if (!usePNGs && colorExists && depthExists) {
                try {
                    // Temporarily open the files using FFmpeg demuxer to query metadata
                    std::unique_ptr<FFmpegDemuxer> demux_color(new FFmpegDemuxer(cam.pathColor.c_str()));
                    std::unique_ptr<FFmpegDemuxer> demux_depth(new FFmpegDemuxer(cam.pathDepth.c_str()));

                    // Check RGB file resolution vs JSON
                    if (demux_color->GetWidth() != cam.res_x || demux_color->GetHeight() != cam.res_y) {
                        std::cout << "[WARNING] Color video resolution (" << demux_color->GetWidth() << "x" << demux_color->GetHeight()
                                  << ") does not match JSON resolution parameters (" << cam.res_x << "x" << cam.res_y << ")." << std::endl;
                        std::cout << " -> Artifact: Viewport misalignment; the rendered frame will be offset or cropped." << std::endl;
                    }

                    // Check Depth file resolution vs JSON
                    if (demux_depth->GetWidth() != cam.res_x || demux_depth->GetHeight() != cam.res_y) {
                        std::cout << "[WARNING] Depth video resolution (" << demux_depth->GetWidth() << "x" << demux_depth->GetHeight()
                                  << ") does not match JSON resolution parameters (" << cam.res_x << "x" << cam.res_y << ")." << std::endl;
                        std::cout << " -> Artifact: Warp texture coordinates will misalign with RGB values, causing halo boundaries." << std::endl;
                    }

                    // Check FPS matching
                    double colorFps = demux_color->GetFPS();
                    double depthFps = demux_depth->GetFPS();
                    if (std::abs(colorFps - depthFps) > 0.5) {
                        std::cout << "[WARNING] Mismatched video framerates: Color runs at " << colorFps << " FPS, but Depth runs at "
                                  << depthFps << " FPS." << std::endl;
                        std::cout << " -> Artifact: Temporal desynchronization; depth and color textures will desync, resulting in ghosting shadows during movement." << std::endl;
                    }

                    // Check frame count matching
                    int64_t colorFrames = demux_color->GetFrameCount();
                    int64_t depthFrames = demux_depth->GetFrameCount();
                    if (colorFrames != depthFrames) {
                        std::cout << "[WARNING] Mismatched frame count: Color has " << colorFrames << " frames, Depth has "
                                  << depthFrames << " frames." << std::endl;
                        std::cout << " -> Artifact: The video stream with fewer frames will freeze early or trigger thread pool decoding failures." << std::endl;
                    }

                    // Verify depth bit-depth consistency
                    int actualDepthBitDepth = demux_depth->GetBitDepth();
                    if (actualDepthBitDepth != cam.bitdepth_depth) {
                        std::cout << "[WARNING] JSON specifies depth bit depth as " << cam.bitdepth_depth
                                  << ", but the depth video file is encoded in " << actualDepthBitDepth << "-bit." << std::endl;
                        std::cout << " -> Artifact: Texture allocation mismatch. Pixel values will be interpreted incorrectly, causing depth compression." << std::endl;
                    }

                } catch (const std::exception& e) {
                    std::cout << "[WARNING] FFmpeg metadata query failed: " << e.what() << std::endl;
                }
            }
        }

        std::cout << "\n=============================================" << std::endl;
        if (hasCriticalError) {
            std::cout << " [DatasetValidator] STATUS: FAILED (Critical errors present)" << std::endl;
            std::cout << " Please fix the critical errors listed above before running Open-DIBR." << std::endl;
            std::cout << "=============================================\n" << std::endl;
            return false;
        } else {
            std::cout << " [DatasetValidator] STATUS: PASSED (Warnings may be present)" << std::endl;
            std::cout << "=============================================\n" << std::endl;
            return true;
        }
    }

private:
    static bool FileExists(const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    }
};

#endif // DATASET_VALIDATOR_H
