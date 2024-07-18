#include "Engine.h"

#include <thread>

namespace alkaid {

Engine* Engine::Builder::build() {
	return Engine::create(*this);
}

Engine::Engine(const Builder& builder) {
	
}

Engine* Engine::create(const Builder& builder) {
	return new Engine(builder);
}

int Engine::loop() {
	return 0;
}

}