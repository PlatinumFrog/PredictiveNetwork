#include <iostream>
#include <random>

#include <SDL3/SDL.h>

#include "Chat.h"

constexpr float step = 1.0f/256.0f;

constexpr float range = 1.0f;

constexpr uint32_t viewWidth = 2560;
constexpr uint32_t viewHeight = 1440;

void GLAPIENTRY MessageCallback(
	GLenum source,
	GLenum type,
	unsigned int id,
	GLenum severity,
	GLsizei length,
	const char* message,
	const void* userParam
) {
	// ignore non-significant error/warning codes
	//if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source) {
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type) {
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity) {
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}

int main(int argc, char* argv[]) {

	std::srand(28172);

	SDL_Init(SDL_INIT_CAMERA | SDL_INIT_VIDEO);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 16);
	SDL_GL_SetSwapInterval(-1);
#ifdef _DEBUG
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif // DEBUG

	SDL_Window* window = SDL_CreateWindow("Chat", viewWidth, viewHeight, SDL_WINDOW_FULLSCREEN | SDL_WINDOW_OPENGL);
	SDL_GLContext context = SDL_GL_CreateContext(window);
	gladLoadGL();
#if _DEBUG
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(MessageCallback, 0);
#endif
	glViewport(0, 0, viewWidth, viewHeight);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glCullFace(GL_BACK);
	glEnable(GL_MULTISAMPLE);

	

	const bool* keystates = SDL_GetKeyboardState(NULL);
	bool loop = true;
	while (!keystates[SDL_SCANCODE_ESCAPE] && loop) {
		keystates = SDL_GetKeyboardState(NULL);
		SDL_PumpEvents();
		SDL_Event e;
		std::vector<SDL_Event> events;
		while (SDL_PollEvent(&e)) {
			events.push_back(e);
		}
		for (SDL_Event& ev : events) {
			switch (ev.type) {
			case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
				loop = false;
				break;
			case SDL_EVENT_WINDOW_RESIZED:
				glViewport(0, 0, ev.window.data1, ev.window.data2);
				break;
			default:
				break;
			}
		}

		events.clear();



		std::cout << "Bot: " << c.getOutput() << "\n";

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		c.draw();

		SDL_GL_SwapWindow(window);

	}

	SDL_GL_DestroyContext(context);
	SDL_HideWindow(window);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}