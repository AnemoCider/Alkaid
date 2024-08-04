module;

#include "vulkan/VulkanDriver.h"

#include <thread>
#include <iostream>

module engine;

namespace alkaid {

Engine* Engine::Builder::build() {
	return Engine::create(*this);
}

Engine::Engine(const Builder& builder) {
    
}

Engine* Engine::create(const Builder& builder) {
	Engine* instance = new Engine(builder);

    instance->mDriverThread = std::thread(&Engine::loop, instance);
    instance->mDriverSemaphore.acquire();

    return instance;
}

int Engine::loop() {
	mDriver = vki::Driver::create();
    mDriverSemaphore.release();
    return 0;
}

Engine::~Engine() noexcept {
    mDriverThread.join();
    delete mDriver;
}

}