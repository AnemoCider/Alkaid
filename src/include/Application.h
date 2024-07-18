#pragma once

namespace alkaid {

class Application {
public:
	static Application& get();
	void run();
};

}