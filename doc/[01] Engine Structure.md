# Engine Structure

## CMake

Alkaid uses modern CMake to build.

- Out of source build: never build in the source directory. Use `cmake -S . -B build` to specify the source dir and build dir, then `cmake --build build` to build.
- Visualize dependency: Use `cmake .. --graphviz=test.dot` to generate a dependency graph, which I first learned from here: https://zhuanlan.zhihu.com/p/483436581.

### Debugging

In bash, use `cmake --build build -v` to show log, `cmake build --trace-source="CMakeLists.txt"` to trace line by line.

### Target

- Definition: the executable or library built, specified by `add_executable` or `add_library`. A target is like an "object" in OOP, in that it has properties and methods.
- Linking: `target_link_libraries` describes the relationship between targets.
  - Each target has 2 collections of properties: `private`s control what happens when you build the target, `interface` tells targets linked to this one what to do when building. `PUBLIC` keyword fills both.
  - Example commands: `target_include_directories`, `target_compile_features`, `target_compile_definitions`, `target_compile_options`

### Variables

- Local variables: `set(MY_VARIABLE "I am a variable")`, `message(STATUS "${MY_VARIABLE}")`.
  - child scopes inherit parent scopes, but not the other way around.
  - to go the other way, use `set (VAR "xxx" PARENT_SCOPE)` in the child scope
- Cached variables: set in the command line or a graphic tool, then stored in `CMakeCache.txt`. CMake remembers what you ran it with before starting a rebuild, by reading the cache. You can manually cache a variable by `set(MY_CACHE_VAR "I am a cached variable" CACHE STRING "Description")`. The type of the variable is necessary to help graphical CMake tools show the correct options.
  - Option: `option`s are just shortcuts for cached bool variables. Example: `option(MY_OPTION "On or off" OFF)`. 
  - Override: `cmake -DMY_OPTION=ON ..`
- Env variables: set by `$ENV{name}`. Accessed without `$`, e.g., `if(DEFINED ENV{name})`.
- Properties: variables that are attached to a target. `get_property`

### Globbing

`file(GLOB OUTPUT_VAR *.cxx)`

- Using GLOB is not recommended, and strongly discouraged without `CONFIGURE_DEPENDS`.
- make a list of all files that match the pattern and put them into the variable
- `GLOB_RECURSE` to recurse subdirs
- `CONFIGURE_DEPENDS`: rerun cmake in the build step only as needed. 

### Finding Packages

- For small projects, find packages in the main CMakeLists, then use them in subdirectories.