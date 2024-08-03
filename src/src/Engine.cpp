module;
#include "vulkan/VulkanInstance.h"

#include <thread>

export module engine;

export namespace alkaid {

class Engine {
public:
	class Builder {
	public:
		Builder() = default;
		Engine* build();
	};

	Engine(const Builder& builder);

	int loop();

private:
	static Engine* create(const Builder& builder);
};

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