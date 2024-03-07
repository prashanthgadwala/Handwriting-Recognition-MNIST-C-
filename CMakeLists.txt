cmake_minimum_required(VERSION 3.26)
project (ws2023-group-15-idkcoding)
set (CMAKE_CXX_STANDARD 23)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
include_directories(.)
include_directories(src, Tensor)
add_executable(ws2023-group-15-idkcoding
    src/matvec.hpp
    src/tensor.hpp
    src/test.cpp
    src/neural_net.hpp
    src/train.cpp)
target_link_libraries(ws2023-group-15-idkcoding "${TORCH_LIBRARIES}")
add_executable(train
    src/train.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}")
add_executable(test
    src/test.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")