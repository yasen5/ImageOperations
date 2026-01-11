cmake_minimum_required(VERSION 3.29)
project(ImageGradients)

if (MSVC)
    add_compile_options(/GR)      # Enable RTTI for MSVC
else ()
    add_compile_options(-frtti)   # Enable RTTI for GCC/Clang
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_DEBUG "-g")

find_package(Eigen3 5.0 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

add_subdirectory(SimpleEdgeDetection)

add_executable(ImageGradients main.cpp)
target_link_libraries(ImageGradients EdgeDetection ${OpenCV_LIBS})
target_include_directories(ImageGradients PRIVATE ${OpenCV_INCLUDE_DIRS})
