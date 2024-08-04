module;

#include "vulkan/VulkanDriver.h"

#include <thread>
#include <semaphore>

export module engine;

namespace alkaid {

export class Engine {
public:
	class Builder {
	public:
		Builder() = default;
		Engine* build();
	};

	Engine(const Builder& builder);

    ~Engine() noexcept;

private:
    std::thread mDriverThread;
    std::binary_semaphore mDriverSemaphore{0};

    vki::Driver* mDriver = nullptr;

	static Engine* create(const Builder& builder);
    int loop();
};

}