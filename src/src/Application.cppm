export module application;

namespace alkaid {
    export class Application {
    public:
        static Application& get();
        void run();
    };

    Application& Application::get() {
        static Application application;
        return application;
    }

    void Application::run() {
        
    }
}




