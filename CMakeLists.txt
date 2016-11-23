# @date August 2016
# @author Alex Giokas <alexge233@hotmail.com>
# 
PROJECT(relearn)
set (RAPP_VERSION_MAJOR 0)
set (RAPP_VERSION_MINOR 1)
cmake_minimum_required(VERSION 2.8)
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
set(SRC ${SRC} SRC)
include_directories(${SRC})

# build examples
set(EXAMPLES ${EXAMPLES} examples)
add_executable(gridworld ${EXAMPLES}/gridworld.cpp)
add_executable(blackjack ${EXAMPLES}/blackjack.cpp)


# set output
set(CMAKE_COLOR_MAKEFILE on)
set(CMAKE_VERBOSE_MAKEFILE off)

# set debug
set(CMAKE_BUILD_TYPE Debug)

# set C++14
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++14 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mtune=native -march=native")
message(STATUS "CXX Flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "CXX Linker: " ${CMAKE_EXE_LINKER_FLAGS})
