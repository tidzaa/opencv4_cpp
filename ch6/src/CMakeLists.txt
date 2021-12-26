cmake_minimum_required (VERSION 3.0)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++17 -O3 -fopenmp -pthread")

include_directories(${PROJECT_SOURSE_DIR}/include)

# Requires OpenCV
find_package(OpenCV  REQUIRED)

message("OpenCV version : ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_library(utils utils.cpp)
add_library(mwindow mwindow.cpp)
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} utils mwindow -lopencv_ml)