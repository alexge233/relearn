# @date August 2016
# @author Alex Giokas <alexge233@hotmail.com>
# 
PROJECT(relearn)
set (RAPP_VERSION_MAJOR 0)
set (RAPP_VERSION_MINOR 1)
cmake_minimum_required(VERSION 2.8)
# set C++14
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++1y -Wall")
add_compile_options(-std=c++1y)
add_definitions("-std=c++1y")
if (CMAKE_COMPILER_IS_GNUCXX)
    set (CMAKE_CXX_STANDARD 14)
endif()

# Library paths for linux
set(LIBRARY_PATH ${LIBRARY_PATH}
		      /lib
		      /usr/lib
		      /usr/lib64
		      /usr/local/lib
		      /usr/local/lib64
		      /usr/lib/x86_64-linux-gnu)
set(INCLUDE_HEADERS ${INCLUDE_HEADERS} /usr/include/)
include_directories(SYSTEM ${INCLUDE_HEADERS})
set(CMAKE_MACOSX_RPATH 1)

# include "src"
set(SRC ${SRC} src)
include_directories(${SRC})

# build examples - default behaviour when CMake is invoked
find_package(Boost 1.55 COMPONENTS serialization system)
set(EXAMPLES ${EXAMPLES} examples)
add_executable(ex_gridworld_offline ${EXAMPLES}/gridworld_offline.cpp)
add_executable(ex_gridworld_online  ${EXAMPLES}/gridworld_online.cpp)
add_executable(ex_blackjack         ${EXAMPLES}/blackjack.cpp)
target_link_libraries(ex_blackjack  ${Boost_LIBRARIES})
# TODO: add an example for serialization of policy, state, action
#       which uses the definition: 

# set output
set(CMAKE_COLOR_MAKEFILE on)
set(CMAKE_VERBOSE_MAKEFILE off)
# set Release as default build type
# you may use `-DCMAKE_BUILD_TYPE=Debug` if you wanna change this
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mtune=native -march=native")
message(STATUS "CXX Flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "CXX Linker: " ${CMAKE_EXE_LINKER_FLAGS})

# if building tests
if (BUILD_TESTS)
    message(STATUS "Checking ${PROJECT_SOURCE_DIR}/tests/catch.hpp")
    if (EXISTS "${PROJECT_SOURCE_DIR}/tests/catch.hpp")
        message(STATUS "catch.hpp exists, not downloading again - make sure it is up to date!")
    else()
        set(wget_command "wget")
        set(wget_arg "https://raw.githubusercontent.com/philsquared/Catch/master/single_include/catch.hpp")
        message(STATUS "Fetching catch.hpp single header")
        execute_process(COMMAND ${wget_command} ${wget_arg} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/)
    endif()

    enable_testing()
    add_executable(test_classes tests/class_test.cpp)
    add_test(unit_test test_classes)
    add_executable(test_logic tests/logic_test.cpp)
    add_test(logic_test test_logic)

    # if building tests & testing serialization
    if (USING_BOOST_SERIALIZATION)
        add_definitions(-DUSING_BOOST_SERIALIZATION)
        find_package(Boost 1.55 COMPONENTS serialization system)
        include_directories(${Boost_INCLUDE_DIR})
        add_executable(test_serialize tests/serialize_test.cpp)
        target_link_libraries(test_serialize ${Boost_LIBRARIES})
        add_test(serialize_test test_serialize)
    endif()
endif()
