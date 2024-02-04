#include "VulkanBase.h"

void Base::init() {
	instance.init();
	instance.createWindow(this);

	device.setInstance(&instance);
#ifdef __APPLE__
	device.addExtension("VK_KHR_portability_subset");
#endif

	device.init();
	device.getGraphicsQueue(graphicsQueue);
}

void Base::prepare() {
	swapChain.setDevice(&device);
	swapChain.setInstance(&instance);
	swapChain.init();
}


void Base::clear() {
	swapChain.clear();
	device.clear();
	instance.destroyWindow();
	instance.clear();
}
