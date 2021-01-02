cmake_minimum_required(VERSION 3.8)

set(This NeuralNetwork)

project(${This} C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

add_subdirectory(googletest)

set(NNHeaders
    NeuralNetwork.hpp
    Neuron.hpp
    Layer.hpp
    Matrix.hpp
)

set(NNSources
    NeuralNetwork.cpp
    Neuron.cpp
    Layer.cpp
)

add_library(${This} STATIC ${NNSources} ${NNHeaders})

add_executable (Main Main.cpp)
target_link_libraries(Main PUBLIC NeuralNetwork)

add_subdirectory(test)