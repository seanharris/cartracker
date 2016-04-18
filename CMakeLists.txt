cmake_minimum_required(VERSION 2.8.12)

project(car)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(car main.cpp)
target_link_libraries(car ${OpenCV_LIBS})
