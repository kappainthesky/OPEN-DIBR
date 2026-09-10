 #ifndef PC_APPLICATION_H
#define PC_APPLICATION_H

#include "Application.h"
#include "EyeTrackerReceiver.h"


class PCApplication : public Application
{
public:
	PCApplication(Options options, FpsMonitor* fpsMonitor, std::vector<InputCamera> inputCameras, std::vector<OutputCamera> outputCameras);

	bool BInitGL();
	bool HandleUserInput();

	// keep track of user input
	glm::vec3 accumMovement = glm::vec3();
	glm::vec3 accumRotation = glm::vec3();

private:
	bool useEyeTracking = false;
};

PCApplication::PCApplication(Options options, FpsMonitor* fpsMonitor, std::vector<InputCamera> inputCameras, std::vector<OutputCamera> outputCameras)
	: Application(options, fpsMonitor, inputCameras, outputCameras){};


bool PCApplication::BInitGL()
{
	if (!Application::BInitGL()) {
		return false;
	}
	framebuffers.init(inputCameras, options.SCR_WIDTH, options.SCR_HEIGHT, options);

	useEyeTracking = eyeReceiver.Init(9999);
	if (useEyeTracking) {
		std::cout << "[PCApplication] Screen-calibrated eye tracking receiver initialized on port 9999." << std::endl;
	}
	return true;
}

bool PCApplication::HandleUserInput() 
{
	SDL_Event sdlEvent;
	bool bRet = false;

	glm::vec3 movement(0);
	glm::vec3 rotation(0);

	while (SDL_PollEvent(&sdlEvent) != 0)
	{
		if (sdlEvent.type == SDL_QUIT)
		{
			bRet = true;
		}
		else if (sdlEvent.type == SDL_KEYDOWN)
		{
			if (sdlEvent.key.keysym.sym == SDLK_ESCAPE)
			{
				bRet = true;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_1)
			{
				options.debugMode = 0;
				shaders.shader.use();
				shaders.shader.setFloat("debugMode", 0.0f);
				std::cout << "[Visualizer] Mode 1: Virtual Mirror (Novel View Render)" << std::endl;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_2)
			{
				options.debugMode = 5;
				shaders.shader.use();
				shaders.shader.setFloat("debugMode", 6.0f);
				std::cout << "[Visualizer] Mode 2: Depth Heatmap (Jet Colormap in meters)" << std::endl;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_3)
			{
				options.debugMode = 6;
				shaders.shader.use();
				shaders.shader.setFloat("debugMode", 7.0f);
				std::cout << "[Visualizer] Mode 3: Invalid Depth Mask (Red=Holes/Invalid, Green=Valid)" << std::endl;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_4)
			{
				options.debugMode = 4;
				shaders.shader.use();
				shaders.shader.setFloat("debugMode", 0.0f);
				std::cout << "[Visualizer] Mode 4: Mesh Only (Zero Discontinuity Deletion)" << std::endl;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_f)
			{
				if (options.useRealSenseInput) {
					realSenseReceiver.ToggleDisparity();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_s)
			{
				if (options.useRealSenseInput) {
					realSenseReceiver.ToggleSpatial();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_t)
			{
				if (options.useRealSenseInput) {
					realSenseReceiver.ToggleTemporal();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_h)
			{
				if (options.useRealSenseInput) {
					realSenseReceiver.ToggleHoleFilling();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_v)
			{
				cameraSpeed += cameraSpeed * 0.1f;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_c)
			{
				cameraSpeed = std::max(cameraSpeed * 0.9f, 0.001f);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_n)
			{
				options.blendingFactor = std::min(options.blendingFactor + 1, 10);
				std::cout << "changed blending_factor to " << options.blendingFactor << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("blendingThreshold", 0.001f + options.blendingFactor * 0.004f);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_b)
			{
				options.blendingFactor = std::max(options.blendingFactor -1, 0);
				std::cout << "changed blending_factor to " << options.blendingFactor << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("blendingThreshold", 0.001f + options.blendingFactor * 0.004f);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_UP)
			{
				options.triangle_deletion_margin += (options.useRealSenseInput ? 0.2f : 2.0f);
				std::cout << "Changed triangle_deletion_margin to " << std::fixed << std::setprecision(2) << options.triangle_deletion_margin << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("triangle_deletion_margin", options.triangle_deletion_margin);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_DOWN)
			{
				options.triangle_deletion_margin = std::max(options.triangle_deletion_margin - (options.useRealSenseInput ? 0.2f : 2.0f), 0.2f);
				std::cout << "Changed triangle_deletion_margin to " << std::fixed << std::setprecision(2) << options.triangle_deletion_margin << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("triangle_deletion_margin", options.triangle_deletion_margin);
			}
			else if (options.showCameraVisibilityWindow && sdlEvent.key.keysym.sym == SDLK_r) {
				controlCameraVisibilityWindow = !controlCameraVisibilityWindow;
				if (controlCameraVisibilityWindow) std::cout << "now controlling the small window in the bottom right corner" << std::endl;
				else std::cout << "now controlling the main window" << std::endl;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_k) {
				// Reset manual keyboard/mouse offset without touching hardware screen calibration
				accumMovement = glm::vec3(0.0f);
				accumRotation = glm::vec3(0.0f);
				if (useEyeTracking) {
					eyeReceiver.ResetCalibration();
				}
				std::cout << "[PCApplication] Manual viewpoint offset reset to centered baseline (0, 0, 0)." << std::endl;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_8) {
				if (useEyeTracking) {
					eyeReceiver.ToggleViewpointUpdate();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_9) {
				if (useEyeTracking) {
					eyeReceiver.ToggleXAxis();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_0) {
				if (useEyeTracking) {
					eyeReceiver.ToggleYAxis();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_j) {
				if (useEyeTracking) {
					eyeReceiver.ToggleZAxis();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_l) {
				if (useEyeTracking) {
					eyeReceiver.ToggleInvertZ();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_i) {
				if (useEyeTracking) {
					eyeReceiver.ToggleInvertX();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_o) {
				if (useEyeTracking) {
					eyeReceiver.ToggleInvertY();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_p) {
				if (useEyeTracking) {
					eyeReceiver.TogglePrediction();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_LEFTBRACKET) {
				if (useEyeTracking) {
					eyeReceiver.DecreaseGain();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_RIGHTBRACKET) {
				if (useEyeTracking) {
					eyeReceiver.IncreaseGain();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_MINUS) {
				if (useEyeTracking) {
					eyeReceiver.DecreaseClamp();
				}
			}
			else if (sdlEvent.key.keysym.sym == SDLK_EQUALS) {
				if (useEyeTracking) {
					eyeReceiver.IncreaseClamp();
				}
			}
		}
		else if (sdlEvent.type == SDL_MOUSEBUTTONDOWN && sdlEvent.button.button == SDL_BUTTON_LEFT) {
			leftMouseDown = true;
			prev_mouse_pos_x = (float)sdlEvent.motion.x;
			prev_mouse_pos_y = (float)sdlEvent.motion.y;
		}
		else if (sdlEvent.type == SDL_MOUSEBUTTONUP && sdlEvent.button.button == SDL_BUTTON_LEFT)
		{
			leftMouseDown = false;
		}
		else if (sdlEvent.type == SDL_MOUSEBUTTONDOWN && sdlEvent.button.button == SDL_BUTTON_MIDDLE) {
			middleMouseDown = true;
			prev_mouse_pos_x = (float)sdlEvent.motion.x;
			prev_mouse_pos_y = (float)sdlEvent.motion.y;
		}
		else if (sdlEvent.type == SDL_MOUSEBUTTONUP && sdlEvent.button.button == SDL_BUTTON_MIDDLE)
		{
			middleMouseDown = false;
		}
		else if (leftMouseDown && sdlEvent.type == SDL_MOUSEMOTION)
		{
			float deltaX = (float)sdlEvent.motion.x - prev_mouse_pos_x;
			float deltaY = (float)sdlEvent.motion.y - prev_mouse_pos_y;

			prev_mouse_pos_x = (float)sdlEvent.motion.x;
			prev_mouse_pos_y = (float)sdlEvent.motion.y;

			if (controlCameraVisibilityWindow) {
				cameraVisibilityWindow.angle -= deltaX * 0.005f;
			}
			else {
				// rotation
				rotation.y = -deltaX * 0.002f;
				rotation.x = -deltaY * 0.002f;
			}
		}
		else if (middleMouseDown && sdlEvent.type == SDL_MOUSEMOTION)
		{
			float deltaX = (float)sdlEvent.motion.x - prev_mouse_pos_x;
			float deltaY = (float)sdlEvent.motion.y - prev_mouse_pos_y;

			prev_mouse_pos_x = (float)sdlEvent.motion.x;
			prev_mouse_pos_y = (float)sdlEvent.motion.y;

			if (controlCameraVisibilityWindow) {
				cameraVisibilityWindow.angle -= deltaX * 0.005f;
			}
			else {
				// translation
				movement.x = -deltaX * 0.05f * cameraSpeed;
				movement.y = deltaY * 0.05f * cameraSpeed;
			}
		}
		else if (sdlEvent.type == SDL_MOUSEWHEEL)
		{
			if (sdlEvent.wheel.y > 0) // scroll up
			{
				if (controlCameraVisibilityWindow) {
					cameraVisibilityWindow.radius += 0.05f * abs(cameraVisibilityWindow.radius);
				}
				else {
					// translation
					movement.z -= cameraSpeed * 6;
				}
			}
			else if (sdlEvent.wheel.y < 0) // scroll down
			{
				if (controlCameraVisibilityWindow) {
					cameraVisibilityWindow.radius -= 0.05f * abs(cameraVisibilityWindow.radius);
				}
				else {
					// translation
					movement.z += cameraSpeed * 6;
				}
			}
		}
	}

	const Uint8* state = SDL_GetKeyboardState(NULL);
	if (state[SDL_SCANCODE_W]) { // UP
		movement.y += cameraSpeed;
	}
	if (state[SDL_SCANCODE_S]) { // DOWN
		movement.y -= cameraSpeed;
	}
	if (state[SDL_SCANCODE_D]) { // RIGHT
		movement.x += cameraSpeed;
	}
	if (state[SDL_SCANCODE_A]) { // LEFT
		movement.x -= cameraSpeed;
	}
	if (state[SDL_SCANCODE_Z]) { // BACKWARD
		movement.z += cameraSpeed;
	}
	if (state[SDL_SCANCODE_Q]) { // FORWARD
		movement.z -= cameraSpeed;
	}

	// Check eye tracker updates
	float dx = 0.0f, dy = 0.0f, dz = 0.0f;
	bool gotNewOffset = false;
	if (useEyeTracking) {
		gotNewOffset = eyeReceiver.GetLatestEyeOffset(dx, dy, dz, 1.0f, 1.0f, 1.0f);
	}

	if (useEyeTracking) {
		int stateVal = eyeReceiver.GetLatestState();
		static auto lostStartTime = std::chrono::steady_clock::now();
		static bool wasTracking = false;

		if (stateVal == 0) { // TRACKING
			wasTracking = true;
			if (gotNewOffset) {
				eyeOffset = glm::vec3(dx, dy, dz);
			}
		}
		else if (stateVal == 1) { // LOW_CONFIDENCE
			// Hold last valid eyeOffset, do not reset
		}
		else if (stateVal == 2) { // LOST
			if (wasTracking) {
				wasTracking = false;
				lostStartTime = std::chrono::steady_clock::now();
			}
			auto lostMs = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - lostStartTime).count();
			
			// Grace period: hold the last valid viewpoint for 500 ms
			if (lostMs > 500) {
				// Only after 500ms of prolonged loss, gently decay back to center
				eyeOffset = glm::mix(eyeOffset, glm::vec3(0.0f), 0.02f);
			}
		}
	}

	accumRotation += rotation;
	glm::mat4 inputRx = glm::rotate(glm::mat4(1.0f), accumRotation[0], glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 inputRy = glm::rotate(glm::mat4(1.0f), accumRotation[1], glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 inputRz = glm::rotate(glm::mat4(1.0f), accumRotation[2], glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 rotMat = inputRz * inputRy * inputRx;
	accumMovement += glm::mat3(pcOutputCamera.startRotMat * rotMat) * movement;

	// 1. Base OpenDIBR viewpoint (original camera position from configuration + manual keyboard/mouse navigation)
	glm::vec3 baseViewpoint = pcOutputCamera.pos + accumMovement;

	// 2. Screen-calibrated eye-tracking offset (treated as a small offset relative to base viewpoint)
	glm::vec3 rotatedEyeOffset = glm::mat3(pcOutputCamera.startRotMat * rotMat) * eyeOffset;

	// 3. Final viewpoint = Base Viewpoint + Eye Offset
	glm::vec3 finalViewpoint = baseViewpoint + rotatedEyeOffset;

	glm::mat4 posMat = glm::translate(glm::mat4(1.0f), finalViewpoint);
	pcOutputCamera.model = posMat * pcOutputCamera.startRotMat * rotMat;
	pcOutputCamera.view = glm::inverse(pcOutputCamera.model);

	// 4. Preserve native OpenDIBR pinhole projection (prevent off-axis mesh stretching)
	pcOutputCamera.useOffAxis = false;

	// Concise runtime diagnostics
	static auto lastDiagTime = std::chrono::steady_clock::now();
	static int diagFrameCount = 0;
	diagFrameCount++;
	auto nowDiag = std::chrono::steady_clock::now();
	float elapsedDiag = std::chrono::duration<float, std::milli>(nowDiag - lastDiagTime).count();
	if (elapsedDiag >= 1000.0f) {
		float curFps = (diagFrameCount * 1000.0f) / elapsedDiag;
		diagFrameCount = 0;
		lastDiagTime = nowDiag;

		float rawX = 0, rawY = 0, rawZ = 0;
		float convX = 0, convY = 0, convZ = 0;
		bool vpEnabled = true, xEnabled = true, yEnabled = true, zEnabled = false, xInv = false, yInv = false, predEnabled = true;
		float trackFps = 30.0f, mpTimeMs = 0.0f;
		eyeReceiver.GetDetailedTelemetry(rawX, rawY, rawZ, convX, convY, convZ, vpEnabled, xEnabled, yEnabled, zEnabled, xInv, yInv, predEnabled, trackFps, mpTimeMs);
		int stateVal = eyeReceiver.GetLatestState();

		std::cout << "\n------------------------------------------------------------\n"
		          << "[EyeTracker Telemetry | Frame #" << currentVideoFrame << " | " << (stateVal == 0 ? "TRACKING" : (stateVal == 1 ? "LOW_CONF" : "LOST")) << "]\n"
		          << "  Eye raw (mm)     : X = " << std::fixed << std::setprecision(1) << rawX << ", Y = " << rawY << ", Z = " << rawZ << " mm\n"
		          << "  Converted (m)    : X = " << std::setprecision(4) << convX << ", Y = " << convY << ", Z = " << convZ << " m\n"
		          << "  Base viewpoint   : X = " << std::setprecision(3) << baseViewpoint.x << ", Y = " << baseViewpoint.y << ", Z = " << baseViewpoint.z << " m\n"
		          << "  Final viewpoint  : X = " << std::setprecision(3) << finalViewpoint.x << ", Y = " << finalViewpoint.y << ", Z = " << finalViewpoint.z << " m\n"
		          << "  Performance      : Tracker FPS: " << std::setprecision(1) << trackFps << " | MediaPipe: " << mpTimeMs << " ms | Render FPS: " << curFps << "\n"
		          << "  Tuning           : Gain = " << std::setprecision(2) << eyeReceiver.GetGain() << "x ([/]) | Clamp = +-" << std::setprecision(2) << eyeReceiver.GetClamp() << " m (-/=)\n"
		          << "  Controls         : Viewpoint=" << (vpEnabled ? "ON" : "OFF [8]") 
		          << " | X=" << (xEnabled ? (xInv ? "INV(-X) [9/I]" : "ON(+X) [9/I]") : "OFF [9]")
		          << " | Y=" << (yEnabled ? (yInv ? "INV(+Y) [0/O]" : "ON(-Y) [0/O]") : "OFF [0]")
		          << " | 25ms Pred=" << (predEnabled ? "ON [P]" : "OFF [P]") << "\n"
		          << "------------------------------------------------------------" << std::endl;
	}

	return bRet;
}


#endif PC_APPLICATION_H