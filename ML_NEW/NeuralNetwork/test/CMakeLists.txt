cmake_minimum_required(VERSION 3.8)

set(This ML_Tests)

set(Sources
    Matrix_Tests.cpp
    Neuron_Tests.cpp
    Layer_Tests.cpp
    NN_Tests.cpp
)

add_executable(${This} ${Sources})
target_link_libraries(${This} PUBLIC
    gtest_main
    NeuralNetwork
)

#       TESTS

add_test(
    NAME Matrix_Tests
    COMMAND ${This}
)

add_test(
    NAME Neuron_Tests
    COMMAND ${This}
)

add_test(
    NAME Layer_Tests
    COMMAND ${This}
)

add_test(
    NAME NN_Tests
    COMMAND ${This}
)
