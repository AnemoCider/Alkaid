#include "VulkanBase.h"

void Base::init() {
	instance.init();
	instance.createWindow(this);
	device.setInstance(&instance);
#ifdef __APPLE__
	device.addExtension("VK_KHR_portability_subset");
#endif
	device.init();
}


void Base::clear() {
	device.clear();
	instance.destroyWindow();
	instance.clear();
}
