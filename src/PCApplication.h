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
		std::cout << "[PCApplication] Eye tracking receiver initialized on port 9999. Press 'K' to recalibrate head center." << std::endl;
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
			else if (sdlEvent.key.keysym.sym == SDLK_h)
			{
				options.triangle_deletion_margin += 2;
				std::cout << "changed triangle_deletion_margin to " << options.triangle_deletion_margin << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("triangle_deletion_margin", options.triangle_deletion_margin);
			}
			else if (sdlEvent.key.keysym.sym == SDLK_g)
			{
				options.triangle_deletion_margin = std::max(options.triangle_deletion_margin - 2, 1.0f);
				std::cout << "changed triangle_deletion_margin to " << options.triangle_deletion_margin << std::endl;
				shaders.shader.use();
				shaders.shader.setFloat("triangle_deletion_margin", options.triangle_deletion_margin);
			}
			else if (options.showCameraVisibilityWindow && sdlEvent.key.keysym.sym == SDLK_r) {
				controlCameraVisibilityWindow = !controlCameraVisibilityWindow;
				if (controlCameraVisibilityWindow) std::cout << "now controlling the small window in the bottom right corner" << std::endl;
				else std::cout << "now controlling the main window" << std::endl;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_k) {
				if (useEyeTracking) {
					eyeReceiver.ResetCalibration();
					std::cout << "[PCApplication] Resetting eye tracking calibration..." << std::endl;
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
		if (stateVal == 0) { // TRACKING
			if (gotNewOffset) {
				eyeOffset = glm::vec3(dx, dy, dz);
			}
		}
		else if (stateVal == 1) { // LOW_CONFIDENCE
			// Freeze: keep last stable eyeOffset, do not update it.
		}
		else if (stateVal == 2) { // LOST
			// Interpolate back to center (0, 0, 0)
			eyeOffset = glm::mix(eyeOffset, glm::vec3(0.0f), 0.05f); // 5% per frame for smooth return
		}
		else if (stateVal == 3) { // RECALIBRATING
			// Wait: interpolate back to center during recalibration
			eyeOffset = glm::mix(eyeOffset, glm::vec3(0.0f), 0.1f);
		}
	}

	accumRotation += rotation;
	glm::mat4 inputRx = glm::rotate(glm::mat4(1.0f), accumRotation[0], glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 inputRy = glm::rotate(glm::mat4(1.0f), accumRotation[1], glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 inputRz = glm::rotate(glm::mat4(1.0f), accumRotation[2], glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 rotMat = inputRz * inputRy * inputRx;
	accumMovement += glm::mat3(pcOutputCamera.startRotMat * rotMat) * movement;

	// Shift camera locally by eyeOffset
	glm::vec3 rotatedEyeOffset = glm::mat3(pcOutputCamera.startRotMat * rotMat) * eyeOffset;

	glm::mat4 posMat = glm::translate(glm::mat4(1.0f), pcOutputCamera.pos + accumMovement + rotatedEyeOffset);
	pcOutputCamera.model = posMat * pcOutputCamera.startRotMat * rotMat;
	pcOutputCamera.view = glm::inverse(pcOutputCamera.model);

	// Off-Axis Projection Matrix calculation for True Windowed 6DoF
	if (useEyeTracking) {
		float dbaseline = eyeReceiver.GetBaselineDistance(); // baseline distance from eye to screen in meters
		float W = dbaseline * (float)pcOutputCamera.res_x / pcOutputCamera.focal_x;
		float H = dbaseline * (float)pcOutputCamera.res_y / pcOutputCamera.focal_y;

		// eyeOffset is (dx, dy, dz) in meters
		float xe = eyeOffset.x;
		float ye = eyeOffset.y;
		float ze = eyeOffset.z + dbaseline; // eye distance to screen plane

		// Avoid division by zero or negative distance if user gets too close
		if (ze < 0.1f) ze = 0.1f;

		float n = pcOutputCamera.z_near;
		float f = pcOutputCamera.z_far;

		// Scale factor to near plane
		float s = n / ze;

		float l = (-W / 2.0f - xe) * s;
		float r = (W / 2.0f - xe) * s;
		float b = (-H / 2.0f - ye) * s;
		float t = (H / 2.0f - ye) * s;

		// Build the off-axis projection matrix (column-major)
		glm::mat4 proj = glm::mat4(0.0f);
		proj[0][0] = 2.0f * n / (r - l);
		proj[1][1] = 2.0f * n / (t - b);
		proj[2][0] = (r + l) / (r - l);
		proj[2][1] = (t + b) / (t - b);
		proj[2][2] = -(f + n) / (f - n);
		proj[2][3] = -1.0f;
		proj[3][2] = -2.0f * f * n / (f - n);

		pcOutputCamera.projectionLeft = proj;
		pcOutputCamera.useOffAxis = true;
	} else {
		pcOutputCamera.useOffAxis = false;
	}

	return bRet;
}


#endif PC_APPLICATION_H