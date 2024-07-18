#pragma once

namespace alkaid {


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


}