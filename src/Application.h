#ifndef APPLICATION_H
#define APPLICATION_H


#include <SDL.h>
#include <GL/glew.h>
#include <SDL_opengl.h>
#if defined( OSX )
#include <Foundation/Foundation.h>
#include <AppKit/AppKit.h>
#include <OpenGL/glu.h>
// Apple's version of glut.h #undef's APIENTRY, redefine it
#define APIENTRY
#else
#include <GL/glu.h>
#endif
#include <stdio.h>
#include <string>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <algorithm>
#include <thread>
#include <deque>
#include <cuda.h>
#include <cudaGL.h> // CUDA OpenGL interop needed for cuGraphicsGLRegisterImage


#if defined(POSIX)
#include "unistd.h"
#endif

#ifndef _WIN32
#define APIENTRY
#endif

#ifndef _countof
#define _countof(x) (sizeof(x)/sizeof((x)[0]))
#endif


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "FFmpegDemuxer.h"
#include "GStreamerReceiver.h"
#include "RealSenseReceiver.h"
#include "shader.h"
#include "ioHelper.h"
#include "DatasetValidator.h"
#include "AlignmentVerifier.h"
#include "PerformanceLogger.h"
#include "EyeTrackerReceiver.h"
#include "glHelper.h"
#include "Pool.h"
#include "CameraVisibilityHelper.h"
#include "MeasureFPS.h"


class Application
{
public:
	Application(Options options, FpsMonitor* fpsMonitor, std::vector<InputCamera> inputCameras, std::vector<OutputCamera> outputCameras);

	virtual bool BInit();
	virtual bool BInitGL();

	virtual void Shutdown();

	void RunMainLoop();
	virtual bool HandleUserInput();
	virtual bool RenderFrame(bool nextVideoFrame, std::string outputCameraName = "", int frameNr = 0);

	virtual void SetupCameras();
	virtual bool SetupStereoRenderTargets();
	virtual void SetupCompanionWindow();
	void SetupYUV420Textures(int texture_height, int luma_height);
	bool SetupRGBTextures();
	bool SetupRealSenseTextures();
	void SetupCUgraphicsResources();
	bool SetupDecodingPool();

	bool RenderTarget(bool nextVideoFrame);
	virtual void RenderCompanionWindow();
	virtual void RenderScene(int i, bool isFirstInput);

	bool CreateAllShaders(float chroma_offset);
	void SaveCompanionWindowToYUV(int frameNr, std::string filename, bool saveAsPNG = false);

protected:

// SDL bookkeeping
	SDL_Window* m_pCompanionWindow;
	SDL_GLContext m_pContext;

// OpenGL bookkeeping

	GLuint m_unCompanionWindowVAO;
	GLuint m_glCompanionWindowIDVertBuffer;
	GLuint m_glCompanionWindowIDIndexBuffer;
	unsigned int m_uiCompanionWindowIndexSize;
	unsigned int m_uiControllerVertcount;

	struct VertexDataWindow
	{
		glm::vec2 position;
		glm::vec2 texCoord;

		VertexDataWindow(const glm::vec2& pos, const glm::vec2 tex) : position(pos), texCoord(tex) {	}
	};

	uint32_t m_nRenderWidth;
	uint32_t m_nRenderHeight;
	uint32_t companionWindowWidth;
	uint32_t companionWindowHeight;
	uint32_t cameraVisibilityWindowWidth;
	uint32_t cameraVisibilityWindowHeight;

	Options options;
	OutputCamera pcOutputCamera;
	std::vector<InputCamera> inputCameras;
	std::vector<OutputCamera> outputCameras;
	ShaderController shaders;
	FrameBufferController framebuffers;
	CameraVisibilityHelper cameraVisibilityHelper;
	CameraVisibilityWindow cameraVisibilityWindow;
	Pool pool;
	FpsMonitor* fpsMonitor;

	GLuint* textures_color = NULL;
	GLuint* textures_depth = NULL;

	// video decoding
	std::vector<CUgraphicsResource*> glGraphicsResources;
	std::vector<IDemuxer*> demuxers;
	std::vector<NvDecoder*> decoders;
	CUcontext* cuContext = NULL;

	// for rendering
	std::unordered_set<int> current_inputsToUse;
	std::unordered_set<int> next_inputsToUse;
	int currentVideoFrame = 0;
	float cameraSpeed = 0.01f;
	bool controlCameraVisibilityWindow = false;

	float lastFrameTimeMs = 0.0f;
	float lastWarpingTimeMs = 0.0f;
	float lastBlendingTimeMs = 0.0f;

	EyeTrackerReceiver eyeReceiver;
	RealSenseReceiver realSenseReceiver;
	glm::vec3 eyeOffset = glm::vec3(0.0f);

	float smoothedTriangleMargin = 10.0f;
	void AnalyzeDepthFrame(int inputIndex);

	// some user input state
	bool leftMouseDown = false;
	bool middleMouseDown = false;
	float prev_mouse_pos_x = 0;
	float prev_mouse_pos_y = 0;

};

Application::Application(Options options, FpsMonitor* fpsMonitor, std::vector<InputCamera> inputCameras, std::vector<OutputCamera> outputCameras)
	: m_pCompanionWindow(NULL)
	, m_pContext(NULL)
	, options(options)
	, fpsMonitor(fpsMonitor)
	, inputCameras(inputCameras)
	, outputCameras(outputCameras)
	, cameraSpeed(options.cameraSpeed)
	, smoothedTriangleMargin(options.triangle_deletion_margin) {
	cuContext = new CUcontext();
};

bool Application::BInit()
{
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		printf("%s - SDL could not initialize! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	int nWindowPosX = 20;
	int nWindowPosY = 20;
	Uint32 unWindowFlags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	//SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);

	m_pCompanionWindow = SDL_CreateWindow("OpenDIBR", nWindowPosX, nWindowPosY, options.SCR_WIDTH, options.SCR_HEIGHT, unWindowFlags);
	if (m_pCompanionWindow == NULL)
	{
		printf("%s - Window could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	if (m_pContext == NULL)
	{
		printf("%s - OpenGL 4.1 Core Context failed (%s). Retrying with OpenGL 3.3 Core...\n", __FUNCTION__, SDL_GetError());
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
		m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	}
	if (m_pContext == NULL)
	{
		printf("%s - OpenGL 3.3 Core Context failed (%s). Retrying with Compatibility Profile...\n", __FUNCTION__, SDL_GetError());
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
		m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	}
	if (m_pContext == NULL)
	{
		printf("%s - Critical: OpenGL context could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		printf("Note: If running NVIDIA drivers on Linux, check if a driver/kernel update occurred without a system reboot.\n");
		return false;
	}

	glewExperimental = GL_TRUE;
	GLenum nGlewError = glewInit();
	if (nGlewError != GLEW_OK)
	{
		printf("%s - Error initializing GLEW! %s\n", __FUNCTION__, glewGetErrorString(nGlewError));
		return false;
	}
	glGetError(); // to clear the error caused deep in GLEW

	if (SDL_GL_SetSwapInterval(0) < 0) // 0 means vsync off, 1 means vsync on
	{
		printf("%s - Warning: Unable to set VSync! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	std::string strWindowTitle = "OpenDIBR";
	SDL_SetWindowTitle(m_pCompanionWindow, strWindowTitle.c_str());

	if (!BInitGL())
	{
		printf("%s - Unable to initialize OpenGL!\n", __FUNCTION__);
		return false;
	}
	return true;
}

bool Application::BInitGL()
{
	// Run Dataset Validation
	if (!DatasetValidator::Validate(inputCameras, options.usePNGs, options.useGStreamerInput, options.useRealSenseInput)) {
		return false;
	}

	if (options.useRealSenseInput) {
		if (!realSenseReceiver.Start(inputCameras, options.rsFilterConfig)) {
			std::cerr << "[Application] Failed to initialize RealSense camera(s)!" << std::endl;
			return false;
		}

		if (options.viewport.res_x > 0 && options.viewport.res_y > 0) {
			pcOutputCamera = options.viewport;
		} else if (inputCameras.size() > 0) {
			pcOutputCamera.res_x = inputCameras[0].res_x;
			pcOutputCamera.res_y = inputCameras[0].res_y;
			pcOutputCamera.focal_x = inputCameras[0].focal_x;
			pcOutputCamera.focal_y = inputCameras[0].focal_y;
			pcOutputCamera.principal_point_x = inputCameras[0].principal_point_x;
			pcOutputCamera.principal_point_y = inputCameras[0].principal_point_y;
		}

		SetupCameras(); // needs to go first
		SetupStereoRenderTargets();
		if (!CreateAllShaders(0.0f))
			return false;

		if (!SetupRealSenseTextures()) {
			return false;
		}
		SetupCompanionWindow();

		// Capture first frame as warmup
		std::vector<const uint8_t*> pRgbs;
		std::vector<const uint16_t*> pDepths;
		if (realSenseReceiver.CaptureAllFrames(pRgbs, pDepths, 3000)) {
			for (size_t i = 0; i < pRgbs.size() && i < inputCameras.size(); i++) {
				glBindTexture(GL_TEXTURE_2D, textures_color[i]);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, inputCameras[i].res_x, inputCameras[i].res_y, GL_RGB, GL_UNSIGNED_BYTE, pRgbs[i]);
				glBindTexture(GL_TEXTURE_2D, textures_depth[i]);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, inputCameras[i].res_x, inputCameras[i].res_y, GL_RED, GL_UNSIGNED_SHORT, pDepths[i]);
			}
		}
		return true;
	}

	// the chroma data is stored "chroma_offset" rows below the luma data
	int luma_height = inputCameras[0].res_y;
	int luma_height_rounded = ((luma_height + 16 - 1) / 16) * 16; //round luma height up to multiple of 16
	int texture_height = luma_height_rounded + /*chroma height */luma_height / 2;
	float chroma_offset = float(luma_height_rounded - luma_height);
	SetupCameras(); // needs to go first
	SetupStereoRenderTargets();
	if (!CreateAllShaders(chroma_offset))
		return false;

	if (options.usePNGs) {
		if (!SetupRGBTextures()) {
			return false;
		}
	}
	else {
		SetupYUV420Textures(texture_height, luma_height);
	}
	SetupCompanionWindow();
	if (!options.usePNGs) {
		SetupCUgraphicsResources();
		if (!SetupDecodingPool()) {
			return false;
		}
	}

	return true;
}

void Application::Shutdown()
{

	if (m_pContext)
	{
		if (m_unCompanionWindowVAO != 0)
		{
			glDeleteVertexArrays(1, &m_unCompanionWindowVAO);
			glDeleteBuffers(1, &m_glCompanionWindowIDVertBuffer);
			glDeleteBuffers(1, &m_glCompanionWindowIDIndexBuffer);
		}
	}

	if (options.useRealSenseInput) {
		realSenseReceiver.Stop();
	}

	if (options.useGStreamerInput && demuxers.size() >= 2) {
		GStreamerReceiver* rgbRec = dynamic_cast<GStreamerReceiver*>(demuxers[0]);
		GStreamerReceiver* depthRec = dynamic_cast<GStreamerReceiver*>(demuxers[1]);
		if (rgbRec && depthRec) {
			std::cout << "\n=============================================\n";
			std::cout << "    [GStreamer Live Performance Metrics]\n";
			std::cout << "=============================================\n";
			std::cout << "Average RGB PTS Delta     : " << std::fixed << std::setprecision(2) << rgbRec->getAvgPTSDelta() << " ms\n";
			std::cout << "Average Depth PTS Delta   : " << std::fixed << std::setprecision(2) << depthRec->getAvgPTSDelta() << " ms\n";
			std::cout << "RGB Access Units Count    : " << rgbRec->getAppsinkAUCount() << "\n";
			std::cout << "Depth Access Units Count  : " << depthRec->getAppsinkAUCount() << "\n";
			std::cout << "RGB Demux Calls Count     : " << rgbRec->getDemuxCallsCount() << "\n";
			std::cout << "Depth Demux Calls Count   : " << depthRec->getDemuxCallsCount() << "\n";
			std::cout << "Timeout Count (RGB/Depth) : " << rgbRec->getTimeoutCount() << " / " << depthRec->getTimeoutCount() << "\n";
			std::cout << "Rendered Frames Count     : " << currentVideoFrame << "\n";
			double maxDiff = std::abs((rgbRec->getCurrentPTS() - depthRec->getCurrentPTS()) / 1000000.0);
			std::cout << "Max RGB-Depth PTS Diff   : " << std::fixed << std::setprecision(2) << maxDiff << " ms\n";
			std::cout << "=============================================\n\n";
		}
	}

	if (!options.isStatic && !options.useRealSenseInput) {
		pool.cleanup();
	}

	framebuffers.cleanup();

	if (!options.usePNGs && !options.useRealSenseInput) {
		for (auto& glGraphicsResource : glGraphicsResources) {
			ck(cuGraphicsUnregisterResource(*glGraphicsResource));
			delete glGraphicsResource;
		}
		glGraphicsResources.clear();
		for (auto& demuxer : demuxers) {
			delete demuxer;
		}
		demuxers.clear();
		for (auto& decoder : decoders) {
			delete decoder;
		}
		decoders.clear();

		// do this after decoders are cleared
		if (cuContext) {
			ck(cuCtxDestroy(*cuContext));
			delete cuContext;
		}
	}

	if (textures_color != NULL) {
		glDeleteTextures((GLsizei)inputCameras.size(), textures_color);
		delete[] textures_color;
	}
	if (textures_depth != NULL) {
		glDeleteTextures((GLsizei)inputCameras.size(), textures_depth);
		delete[] textures_depth;
	}


	if (m_pCompanionWindow)
	{
		SDL_DestroyWindow(m_pCompanionWindow);
		m_pCompanionWindow = NULL;
	}

	SDL_Quit();
}

bool Application::HandleUserInput()
{
	return false;
}

void Application::RunMainLoop()
{
	bool bQuit = false;

	SDL_StartTextInput();

	int frame = 0;

	if (!options.asap) {
		float ms_per_frame = 1000.0f / (float)options.targetFps;
		int frames_per_video_frame = options.targetFps / 30;
		if (frames_per_video_frame < 1) frames_per_video_frame = 1;

		while (!bQuit)
		{
			Uint64 blockStartTime = SDL_GetPerformanceCounter();

			// Frame 1: Decode and Render
			RenderFrame(true);
			bQuit = bQuit | HandleUserInput();
			SpinUntilTargetTime(blockStartTime, ms_per_frame);

			Uint64 endTime = SDL_GetPerformanceCounter();
			float passedTimeMs = (endTime - blockStartTime) / (float)SDL_GetPerformanceFrequency() * 1000.0f;
			fpsMonitor->AddTime(passedTimeMs, frame);
			float cumulativeTime = passedTimeMs;
			lastFrameTimeMs = passedTimeMs;

			// Remaining Frames in the 30Hz block: Just render with updated head tracking
			for (int i = 1; i < frames_per_video_frame; i++) {
				RenderFrame(false);
				bQuit = bQuit | HandleUserInput();
				SpinUntilTargetTime(blockStartTime, (i + 1) * ms_per_frame);

				endTime = SDL_GetPerformanceCounter();
				passedTimeMs = (endTime - blockStartTime) / (float)SDL_GetPerformanceFrequency() * 1000.0f;
				float diffMs = passedTimeMs - cumulativeTime;
				fpsMonitor->AddTime(diffMs, frame);
				cumulativeTime = passedTimeMs;
				lastFrameTimeMs = diffMs;
			}
			frame++;
		}
	}
	else if (options.saveOutputImages) {
		for (int frame = 0; frame < options.outputNrFrames; frame++) {
			for (int i = 0; i < outputCameras.size(); i++) {
				pcOutputCamera = outputCameras[i];

				cameraVisibilityHelper.init(inputCameras, &pcOutputCamera, options.maxNrInputsUsed);
				current_inputsToUse = cameraVisibilityHelper.updateInputsToUse();
				for (auto& c : current_inputsToUse) {
					next_inputsToUse.insert(c); // deep copy
				}

				RenderFrame(frame > 0 && i == 0, outputCameras[i].name, frame);
			}
		}
	}
	else {
		// decode and play the input videos as fast as possible
		Uint64 startTime = SDL_GetPerformanceCounter();
		while (!bQuit) {
			RenderFrame(true);
			bQuit = bQuit | HandleUserInput();

			Uint64 endTime = SDL_GetPerformanceCounter();
			float passedTimeMs = (endTime - startTime) / (float)SDL_GetPerformanceFrequency() * 1000.0f;
			fpsMonitor->AddTime(passedTimeMs, frame);
			lastFrameTimeMs = passedTimeMs;
			startTime = endTime;
			frame++;
		}
	}

	SDL_StopTextInput();
	PerformanceLogger::SaveToCSV("performance_log.csv");
}

bool Application::RenderFrame(bool nextVideoFrame, std::string outputCameraName, int frameNr)
{
	RenderTarget(nextVideoFrame);
	if (outputCameraName != "") {
		SaveCompanionWindowToYUV(frameNr, outputCameraName);
	}
	
	RenderCompanionWindow();

	// Log frame metrics
	PerformanceLogger::LogFrame(frameNr, lastFrameTimeMs, fpsMonitor->GetFPS(),
	                            pool.lastDecodeTimeMs.load(), lastWarpingTimeMs, lastBlendingTimeMs,
	                            smoothedTriangleMargin, 45.0f);

	// SwapWindow
	{
		SDL_GL_SwapWindow(m_pCompanionWindow);
	}

	// Clear
	{
		// We want to make sure the glFinish waits for the entire present to complete, not just the submission
		// of the command. So, we do a clear here right here so the glFinish will wait fully for the swap.
		glClearColor(options.backgroundColor.r, options.backgroundColor.g, options.backgroundColor.b, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	return true;
}

bool Application::CreateAllShaders(float chroma_offset)
{
	return shaders.init(inputCameras[0], options, m_nRenderWidth, m_nRenderHeight, chroma_offset, pcOutputCamera);
}

void Application::SetupCameras()
{
	pcOutputCamera = options.viewport;

	cameraVisibilityHelper.init(inputCameras, &pcOutputCamera, options.maxNrInputsUsed);
	current_inputsToUse = cameraVisibilityHelper.updateInputsToUse();
	for (auto& c : current_inputsToUse) {
		next_inputsToUse.insert(c); // deep copy
	}
}

bool Application::SetupStereoRenderTargets()
{
	m_nRenderWidth = options.SCR_WIDTH;
	m_nRenderHeight = options.SCR_HEIGHT;

	companionWindowWidth = options.SCR_WIDTH;
	companionWindowHeight = options.SCR_HEIGHT;
	return true;
}

void Application::SetupCompanionWindow()
{
	std::vector<VertexDataWindow> vVerts;

	vVerts.push_back(VertexDataWindow(glm::vec2(-1, -1), glm::vec2(0, 0)));
	vVerts.push_back(VertexDataWindow(glm::vec2(1, -1), glm::vec2(1, 0)));
	vVerts.push_back(VertexDataWindow(glm::vec2(-1, 1), glm::vec2(0, 1)));
	vVerts.push_back(VertexDataWindow(glm::vec2(1, 1), glm::vec2(1, 1)));

	GLushort vIndices[] = { 0, 1, 3,   0, 3, 2 };
	m_uiCompanionWindowIndexSize = _countof(vIndices);

	glGenVertexArrays(1, &m_unCompanionWindowVAO);
	glBindVertexArray(m_unCompanionWindowVAO);

	glGenBuffers(1, &m_glCompanionWindowIDVertBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_glCompanionWindowIDVertBuffer);
	glBufferData(GL_ARRAY_BUFFER, vVerts.size() * sizeof(VertexDataWindow), &vVerts[0], GL_STATIC_DRAW);

	glGenBuffers(1, &m_glCompanionWindowIDIndexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glCompanionWindowIDIndexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_uiCompanionWindowIndexSize * sizeof(GLushort), &vIndices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataWindow), (void*)offsetof(VertexDataWindow, position));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataWindow), (void*)offsetof(VertexDataWindow, texCoord));

	glBindVertexArray(0);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	if (options.showCameraVisibilityWindow) {
		cameraVisibilityWindowWidth = companionWindowWidth / 3;
		cameraVisibilityWindowHeight = companionWindowHeight / 3;
		cameraVisibilityWindow.init(cameraVisibilityWindowWidth, cameraVisibilityWindowHeight, inputCameras);
	}
}

void Application::SetupYUV420Textures(int texture_height, int luma_height) {
	textures_color = new GLuint[inputCameras.size()];
	textures_depth = new GLuint[inputCameras.size()];
	glGenTextures((GLsizei)inputCameras.size(), textures_color);
	glGenTextures((GLsizei)inputCameras.size(), textures_depth);
	for (int i = 0; i < inputCameras.size(); i++) {
		// technically only need #threads * 2 textures
		// but for now, use 2 textures per input
		glBindTexture(GL_TEXTURE_2D, textures_color[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		if (inputCameras[i].bitdepth_color > 8) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, inputCameras[0].res_x, texture_height, 0, GL_RED, GL_UNSIGNED_SHORT, 0);
		}
		else {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, inputCameras[0].res_x, texture_height, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
		}

		glBindTexture(GL_TEXTURE_2D, textures_depth[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		if (inputCameras[i].bitdepth_depth > 8) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, inputCameras[0].res_x, luma_height, 0, GL_RED, GL_UNSIGNED_SHORT, 0);
		}
		else {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, inputCameras[0].res_x, luma_height, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
		}
	}
}

bool Application::SetupRGBTextures() {
	textures_color = new GLuint[inputCameras.size()];
	textures_depth = new GLuint[inputCameras.size()];
	glGenTextures((GLsizei)inputCameras.size(), textures_color);
	glGenTextures((GLsizei)inputCameras.size(), textures_depth);

	int width, height, nrChannels;
	for (int i = 0; i < inputCameras.size(); i++) {
		// technically only need #threads * 2 textures
		// but for now, use 2 textures per input
		glBindTexture(GL_TEXTURE_2D, textures_color[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		if (inputCameras[i].bitdepth_color > 8) {
			unsigned short* data = stbi_load_16(inputCameras[i].pathColor.c_str(), &width, &height, &nrChannels, STBI_rgb);
			if (!data) {
				std::cout << "Error: failed to load texture " << inputCameras[i].pathColor << std::endl;
				return false;
			}
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16, inputCameras[0].res_x, inputCameras[0].res_y, 0, GL_RGB, GL_UNSIGNED_SHORT, data);
			stbi_image_free(data);
		}
		else {
			unsigned char* data = stbi_load(inputCameras[i].pathColor.c_str(), &width, &height, &nrChannels, STBI_rgb);
			if (!data) {
				std::cout << "Error: failed to load texture " << inputCameras[i].pathColor << std::endl;
				return false;
			}
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, inputCameras[0].res_x, inputCameras[0].res_y, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
			stbi_image_free(data);
		}

		glBindTexture(GL_TEXTURE_2D, textures_depth[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		if (inputCameras[i].bitdepth_depth > 8) {
			unsigned short* data = stbi_load_16(inputCameras[i].pathDepth.c_str(), &width, &height, &nrChannels, STBI_grey);
			if (!data) {
				std::cout << "Failed to load texture " << inputCameras[i].pathDepth << std::endl;
				return false;
			}
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, inputCameras[0].res_x, inputCameras[0].res_y, 0, GL_RED, GL_UNSIGNED_SHORT, data);
			stbi_image_free(data);
		}
		else {
			unsigned char* data = stbi_load(inputCameras[i].pathDepth.c_str(), &width, &height, &nrChannels, STBI_grey);
			if (!data) {
				std::cout << "Error: failed to load texture " << inputCameras[i].pathDepth << std::endl;
				return false;
			}
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, inputCameras[0].res_x, inputCameras[0].res_y, 0, GL_RED, GL_UNSIGNED_BYTE, data);
			stbi_image_free(data);
		}
	}
	return true;
}

bool Application::SetupRealSenseTextures() {
	textures_color = new GLuint[inputCameras.size()];
	textures_depth = new GLuint[inputCameras.size()];
	glGenTextures((GLsizei)inputCameras.size(), textures_color);
	glGenTextures((GLsizei)inputCameras.size(), textures_depth);

	for (size_t i = 0; i < inputCameras.size(); i++) {
		glBindTexture(GL_TEXTURE_2D, textures_color[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, inputCameras[i].res_x, inputCameras[i].res_y, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

		glBindTexture(GL_TEXTURE_2D, textures_depth[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, inputCameras[i].res_x, inputCameras[i].res_y, 0, GL_RED, GL_UNSIGNED_SHORT, 0);
	}
	return true;
}

void Application::SetupCUgraphicsResources() {
	ck(cuInit(0));

	CUdevice cuDevice = 0;
	ck(cuDeviceGet(&cuDevice, 0));
	char szDeviceName[80];
	ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
	std::cout << "GPU in use: " << szDeviceName << std::endl;
	ck(cuCtxCreate(cuContext, 0, cuDevice));

	ck(cuCtxSetCurrent(*cuContext));
	for (int i = 0; i < inputCameras.size(); i++) {
		// register OpenGL textures for Cuda interop
		CUgraphicsResource* glGraphicsResource_color = new CUgraphicsResource();
		CUgraphicsResource* glGraphicsResource_depth = new CUgraphicsResource();
		ck(cuGraphicsGLRegisterImage(glGraphicsResource_color, textures_color[i], GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
		ck(cuGraphicsGLRegisterImage(glGraphicsResource_depth, textures_depth[i], GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
		ck(cuGraphicsResourceSetMapFlags(*glGraphicsResource_color, CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
		ck(cuGraphicsResourceSetMapFlags(*glGraphicsResource_depth, CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
		glGraphicsResources.push_back(glGraphicsResource_color);
		glGraphicsResources.push_back(glGraphicsResource_depth);

		// initialize demuxers
		IDemuxer* demuxer_color = nullptr;
		IDemuxer* demuxer_depth = nullptr;
		if (options.useGStreamerInput) {
			int cPort = options.colorPort + 2 * i;
			int dPort = options.depthPort + 2 * i;
			GStreamerReceiver* rColor = new GStreamerReceiver(cPort, 96, "ColorReceiver", options.host);
			GStreamerReceiver* rDepth = new GStreamerReceiver(dPort, 97, "DepthReceiver", options.host);
			rColor->SetPeer(rDepth);
			rDepth->SetPeer(rColor);
			demuxer_color = rColor;
			demuxer_depth = rDepth;
		}
		else {
			demuxer_color = new FFmpegDemuxer(inputCameras[i].pathColor.c_str(), i == 0);
			demuxer_depth = new FFmpegDemuxer(inputCameras[i].pathDepth.c_str());
		}
		demuxers.push_back(demuxer_color);
		demuxers.push_back(demuxer_depth);

		// initalize the Cuda Decoders
		NvDecoder* decoder_color = new NvDecoder(cuContext, glGraphicsResource_color, true, FFmpeg2NvCodecId(demuxer_color->GetVideoCodec()), i == 0);
		NvDecoder* decoder_depth = new NvDecoder(cuContext, glGraphicsResource_depth, false, FFmpeg2NvCodecId(demuxer_depth->GetVideoCodec()));
		decoders.push_back(decoder_color);
		decoders.push_back(decoder_depth);
	}
	ck(cuCtxPopCurrent(NULL));
}

bool Application::SetupDecodingPool() {

	if (options.StartingFrameNr > 0) {
		std::cout << "Decoding all frames up until frame " << options.StartingFrameNr << "..." << std::endl;
	}
	// decode until frame 'StartingFrameNr' of all input videos here (in lockstep across streams, 4 frames warmup for 12-bit RExt NVDEC latency)
	for (int j = 0; j < options.StartingFrameNr + 4; j++) {
		for (int i = 0; i < demuxers.size(); i++) {
			int nVideoBytes = 0;
			uint8_t* pVideo = NULL;
			if (!demuxers[i]->Demux(&pVideo, &nVideoBytes)) {
				std::cout << "Error: demuxing failed for input " << i / 2 << (i % 2 == 0 ? " color" : " depth") << std::endl;
				return false;
			}
			decoders[i]->Decode(pVideo, nVideoBytes);
		}
	}
	for (int i = 0; i < demuxers.size(); i++) {
		std::cout << "[Startup] Demuxer " << i << " (" << (i % 2 == 0 ? "color" : "depth") 
		          << ") final picture_index=" << decoders[i]->picture_index << std::endl;
		// memcopy decoded image to CUGragpicsResources
		decoders[i]->HandlePictureDisplay(decoders[i]->picture_index);
	}

	if (!options.isStatic) {
		// setup thread pool to parallelize the decoding work
		pool.init((int)inputCameras.size(), demuxers, decoders, options.nrThreads);
		pool.startThreadPool();
		pool.startDemuxingFirstFrames(current_inputsToUse);
	}
		
	return true;
}

bool Application::RenderTarget(bool nextVideoFrame)
{
	glEnable(GL_DEPTH_TEST);
	glViewport(0, 0, m_nRenderWidth, m_nRenderHeight);

	if (options.useRealSenseInput) {
		if (nextVideoFrame) {
			std::vector<const uint8_t*> pRgbs;
			std::vector<const uint16_t*> pDepths;
			if (realSenseReceiver.CaptureAllFrames(pRgbs, pDepths, 2000)) {
				for (size_t i = 0; i < pRgbs.size() && i < inputCameras.size(); i++) {
					if (pRgbs[i] && pDepths[i]) {
						glBindTexture(GL_TEXTURE_2D, textures_color[i]);
						glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, inputCameras[i].res_x, inputCameras[i].res_y, GL_RGB, GL_UNSIGNED_BYTE, pRgbs[i]);
						glBindTexture(GL_TEXTURE_2D, textures_depth[i]);
						glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, inputCameras[i].res_x, inputCameras[i].res_y, GL_RED, GL_UNSIGNED_SHORT, pDepths[i]);
					}
				}
			}
		}

		bool isFirstInput = true;
		shaders.updateOutputParams(pcOutputCamera);
		shaders.shader.setFloat("debugMode", (float)options.debugMode);

		size_t startIdx = 0;
		size_t endIdx = inputCameras.size();
		if (options.debugMode == 1) { // Left camera only
			startIdx = 0;
			endIdx = std::min((size_t)1, inputCameras.size());
		} else if (options.debugMode == 2) { // Right camera only
			startIdx = std::min((size_t)1, inputCameras.size() - 1);
			endIdx = inputCameras.size();
		}

		for (size_t i = startIdx; i < endIdx; i++) {
			shaders.shader.setFloat("isFirstInput", isFirstInput ? 1.0f : 0.0f);
			shaders.updateInputParams(inputCameras[i]);

			if (options.autoTriangleMargin) {
				AnalyzeDepthFrame((int)i);
			}

			if (currentVideoFrame == 0 || currentVideoFrame == 30) {
				AlignmentVerifier::Verify(textures_color[i], textures_depth[i],
				                          inputCameras[i].res_x, inputCameras[i].res_y,
				                          inputCameras[i].bitdepth_depth,
				                          inputCameras[i].z_near, inputCameras[i].z_far,
				                          currentVideoFrame, (int)i, inputCameras[i].role);
			}

			RenderScene((int)i, isFirstInput);

			if (isFirstInput) {
				isFirstInput = false;
			}
		}

		if (nextVideoFrame) {
			currentVideoFrame++;
		}
		return true;
	}

	bool shouldUpdateUsedInputs = false;
	if (nextVideoFrame) {
		// recalculate which inputCameras need to be used for rendering the outputCamera
		next_inputsToUse = cameraVisibilityHelper.updateInputsToUse();
		for (auto& a : next_inputsToUse) {
			if (current_inputsToUse.find(a) == current_inputsToUse.end()) {
				shouldUpdateUsedInputs = true;
				break;
			}
		}
	}

	bool isFirstInput = true;
	shaders.updateOutputParams(pcOutputCamera);
	shaders.shader.setFloat("isFirstInput", 1.0f);
	for (int i = 0; i < inputCameras.size(); i++) {

		if ((!options.isStatic) && nextVideoFrame) {
			bool useForRenderingNextFrame = next_inputsToUse.find(i) != next_inputsToUse.end();
			pool.startDemuxingNextFrame(i, currentVideoFrame + 1, useForRenderingNextFrame);
		}
		bool useForRenderingCurrentFrame = current_inputsToUse.find(i) != current_inputsToUse.end();

		if (useForRenderingCurrentFrame) {

			shaders.updateInputParams(inputCameras[i]);

			if ((!options.isStatic) && nextVideoFrame) {
				std::tuple<int, int, int, int> tuple = pool.waitUntilInputFrameIsDecoded(i);
				pool.copyFromGPUToOpenGLTexture(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), std::get<3>(tuple));
			}

			if (options.autoTriangleMargin) {
				AnalyzeDepthFrame(i);
			}

			if (isFirstInput) {
				AlignmentVerifier::Verify(textures_color[i], textures_depth[i],
				                          inputCameras[i].res_x, inputCameras[i].res_y,
				                          inputCameras[i].bitdepth_depth,
				                          inputCameras[i].z_near, inputCameras[i].z_far,
				                          currentVideoFrame);
			}

			RenderScene(i, isFirstInput);

			// prepare next iteration
			if (isFirstInput) {
				isFirstInput = false;
				shaders.shader.setFloat("isFirstInput", 0.0f);
			}
		}
	}

	if (nextVideoFrame) {
		currentVideoFrame++; // important for Pool
	}

	if (shouldUpdateUsedInputs) {
		current_inputsToUse.clear();
		for (auto& c : next_inputsToUse) {
			current_inputsToUse.insert(c); // deep copy
		}
	}

	return true;
}

void Application::RenderScene(int i, bool isFirstInput)
{
	if (isFirstInput) {
		auto t_start = std::chrono::high_resolution_clock::now();
		// simple 3D warping
		framebuffers.renderTheFirstInputImage(0, textures_color[i], textures_depth[i]);
		glFinish();
		auto t_end = std::chrono::high_resolution_clock::now();
		lastWarpingTimeMs = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	}
	else {
		auto t_start = std::chrono::high_resolution_clock::now();
		// copying between FBOs is necessary to prepare the blending
		shaders.copyShader.use();
		framebuffers.copyFramebuffer(0);

		// simple 3D warping + blending with the previous output image
		shaders.shader.use();
		framebuffers.renderNonFirstInputImage(0, textures_color[i], textures_depth[i]);
		glFinish();
		auto t_end = std::chrono::high_resolution_clock::now();
		lastBlendingTimeMs = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	}
}

void Application::RenderCompanionWindow()
{
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, options.SCR_WIDTH, options.SCR_HEIGHT);

	glBindVertexArray(m_unCompanionWindowVAO);
	shaders.companionWindowShader.use();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, framebuffers.getColorTexture(0));
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glDrawElements(GL_TRIANGLES, m_uiCompanionWindowIndexSize, GL_UNSIGNED_SHORT, 0);

	// draw input cameras
	if (options.showCameraVisibilityWindow) {
		glViewport(m_nRenderWidth - cameraVisibilityWindowWidth, 0, cameraVisibilityWindowWidth, cameraVisibilityWindowHeight);
		glEnable(GL_SCISSOR_TEST);
		glScissor(m_nRenderWidth - cameraVisibilityWindowWidth, 0, cameraVisibilityWindowWidth, cameraVisibilityWindowHeight);
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDisable(GL_SCISSOR_TEST);
		glm::mat4 view_project = cameraVisibilityWindow.ViewProject();
		shaders.cameraVisibilityShader.use();
		shaders.cameraVisibilityShader.setMat4("view_project", view_project);
		for (int i = 0; i < inputCameras.size(); i++) {
			shaders.cameraVisibilityShader.setVec3("color", (current_inputsToUse.find(i) != current_inputsToUse.end()) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0));
			framebuffers.drawInputCamera(i);
		}
		shaders.cameraVisibilityShader.setMat4("view_project", view_project * pcOutputCamera.model);
		shaders.cameraVisibilityShader.setVec3("color", glm::vec3(0, 1, 1));
		framebuffers.drawOutputCamera();
	}
}

void Application::SaveCompanionWindowToYUV(int frameNr, std::string outputCameraName, bool saveAsPNG) {
	unsigned char* image = new unsigned char[options.SCR_WIDTH * options.SCR_HEIGHT * 4];
	framebuffers.bindCurrentBuffer();
	glReadPixels(0, 0, options.SCR_WIDTH, options.SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, image);
	if (saveAsPNG || options.usePNGs) {
		saveImage(image, options.SCR_WIDTH, options.SCR_HEIGHT, true, frameNr, options.outputPath + outputCameraName + ".png");
	}
	else {
		saveImage(image, options.SCR_WIDTH, options.SCR_HEIGHT, false, frameNr, options.outputPath + outputCameraName + ".yuv");
	}
	delete[] image;
	return;
}

void Application::AnalyzeDepthFrame(int inputIndex) {
	int width = inputCameras[inputIndex].res_x;
	int height = inputCameras[inputIndex].res_y;
	int bitdepth = inputCameras[inputIndex].bitdepth_depth;
	float z_near = inputCameras[inputIndex].z_near;
	float z_far = inputCameras[inputIndex].z_far;

	// Bind the depth texture and download it to CPU
	glBindTexture(GL_TEXTURE_2D, textures_depth[inputIndex]);
	int pixelCount = width * height;
	std::vector<float> depthMeters(pixelCount, 0.0f);

	if (bitdepth > 8) {
		std::vector<uint16_t> buffer(pixelCount);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_SHORT, buffer.data());
		
		for (int idx = 0; idx < pixelCount; ++idx) {
			float normVal = (float)buffer[idx] / 65535.0f;
			if (normVal > 0.0f) {
				depthMeters[idx] = 1.0f / (1.0f / z_far + normVal * (1.0f / z_near - 1.0f / z_far));
			}
		}
	} else {
		std::vector<uint8_t> buffer(pixelCount);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, buffer.data());
		
		for (int idx = 0; idx < pixelCount; ++idx) {
			float normVal = (float)buffer[idx] / 255.0f;
			if (normVal > 0.0f) {
				depthMeters[idx] = 1.0f / (1.0f / z_far + normVal * (1.0f / z_near - 1.0f / z_far));
			}
		}
	}

	// Downsample for analysis to keep CPU overhead minimal (stride of 8 pixels)
	int stride = 8;
	double sumGrad = 0.0;
	int gradCount = 0;
	int invalidCount = 0;
	int totalSampled = 0;

	for (int y = 0; y < height - 1; y += stride) {
		for (int x = 0; x < width - 1; x += stride) {
			int idx = y * width + x;
			float z_curr = depthMeters[idx];
			totalSampled++;

			if (z_curr <= 0.0f || z_curr >= 999.0f) {
				invalidCount++;
				continue;
			}

			// Horizontal gradient
			float z_right = depthMeters[idx + 1];
			if (z_right > 0.0f && z_right < 999.0f) {
				float diff_x = std::abs(z_right - z_curr);
				// To isolate sensor noise and compression block artifacts,
				// we only sum gradients in local areas that are relatively flat (< 0.15m gradient).
				if (diff_x < 0.15f) {
					sumGrad += diff_x;
					gradCount++;
				}
			}

			// Vertical gradient
			float z_bottom = depthMeters[idx + width];
			if (z_bottom > 0.0f && z_bottom < 999.0f) {
				float diff_y = std::abs(z_bottom - z_curr);
				if (diff_y < 0.15f) {
					sumGrad += diff_y;
					gradCount++;
				}
			}
		}
	}

	float invalidRatio = (totalSampled > 0) ? (float)invalidCount / totalSampled : 0.0f;
	float avgGradFlat = (gradCount > 0) ? (float)(sumGrad / gradCount) : 0.0f;

	float estimatedMargin = 10.0f;

	if (options.useRealSenseInput) {
		// Calibrated physical metric depth auto-margin: optimal range [4.0f, 12.0f]
		estimatedMargin = 5.0f + 150.0f * avgGradFlat;
		estimatedMargin *= (1.0f + 1.0f * invalidRatio);
		if (estimatedMargin < 4.0f) estimatedMargin = 4.0f;
		if (estimatedMargin > 12.0f) estimatedMargin = 12.0f;
	} else {
		// Legacy 8-bit dataset formula: safe margins [5.0f, 300.0f]
		estimatedMargin = 10.0f + 6500.0f * avgGradFlat;
		estimatedMargin *= (1.0f + 0.8f * invalidRatio);
		if (estimatedMargin < 5.0f) estimatedMargin = 5.0f;
		if (estimatedMargin > 300.0f) estimatedMargin = 300.0f;
	}

	// Exponential Moving Average (EMA) to prevent screen flickering
	float alpha = 0.08f;
	smoothedTriangleMargin = alpha * estimatedMargin + (1.0f - alpha) * smoothedTriangleMargin;

	// Update the geometry shader uniform
	shaders.shader.use();
	shaders.shader.setFloat("triangle_deletion_margin", smoothedTriangleMargin);
}

#endif APPLICATION_H
