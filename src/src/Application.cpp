#include "Application.h"

using namespace alkaid;

Application& Application::get() {
	static Application application;
	return application;
}

void Application::run() {
	
}
