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

	int loop();

private:
	static Engine* create(const Builder& builder);
};

}