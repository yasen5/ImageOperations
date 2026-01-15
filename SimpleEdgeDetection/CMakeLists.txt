add_library(EdgeDetection kernel.cpp
        canny.cpp
        canny.h)
target_link_libraries(EdgeDetection Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(EdgeDetection PRIVATE ${OpenCV_INCLUDE_DIRS})

