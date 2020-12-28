/**
 * @file NN_Tests.cpp
 * @author Brodie Collinson Davison
 * @brief Test cases for Neural network
 * @version 0.1
 * @date 2020-11-29
 */

#include <gtest/gtest.h>
#include "../NeuralNetwork.hpp"

using namespace NN;

TEST ( NN_Tests, Weight_Matrix_Structure )
{
    int layerSizes [] =
    {
        1, 2, 3, 1
    };

    NeuralNetwork net ( 4, layerSizes );

    for ( int i = 1; i < net.getNumLayers(); i ++ )
    {
        Matrix <float> mat = net.getWeightMatrix ( i );
        cout << "Weight Matrix: " << i << endl;
        mat.print ();
    }
}
