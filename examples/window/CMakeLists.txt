# The window: the watched examples drawn with raylib, as win_<game>, off by default
# (cmake .. -DBUILD_WINDOW=On). raylib is the only dependency, found through its CMake config (Homebrew,
# vcpkg, a source build) or, failing that, as a plain library and header.
find_package(raylib QUIET)
if (NOT raylib_FOUND)
    find_path(RAYLIB_INCLUDE_DIR raylib.h)
    find_library(RAYLIB_LIBRARY NAMES raylib)
    if (NOT RAYLIB_INCLUDE_DIR OR NOT RAYLIB_LIBRARY)
        message(FATAL_ERROR "raylib was not found: install it (brew install raylib, or from source: see docs/watch.md) or pass -DCMAKE_PREFIX_PATH")
    endif()
    add_library(raylib UNKNOWN IMPORTED)
    set_target_properties(raylib PROPERTIES IMPORTED_LOCATION "${RAYLIB_LIBRARY}"
                                            INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_INCLUDE_DIR}")
endif()

# the examples that take --watch, each built again with the window as its screen
set(WINDOW_GAMES cliff_walking windy_gridworld dyna_maze frozen_lake racetrack mountain_car cart_pole)
foreach(game ${WINDOW_GAMES})
    add_executable(win_${game} ${PROJECT_SOURCE_DIR}/examples/${game}.cpp)
    target_compile_definitions(win_${game} PRIVATE RELEARN_WINDOW)
    target_include_directories(win_${game} PRIVATE ${PROJECT_SOURCE_DIR}/examples)
    # next to the ex_<game> programs, in the top of the build directory
    set_target_properties(win_${game} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    target_link_libraries(win_${game} raylib)
    if (APPLE)
        target_link_libraries(win_${game} "-framework IOKit" "-framework Cocoa" "-framework OpenGL")
    endif()
endforeach()
