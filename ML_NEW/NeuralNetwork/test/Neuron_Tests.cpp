/**
 * @file    Neuron_Tests.cpp
 * @author  Brodie Collinson Davison
 * @brief   Tests for Neuron class
 * @version 0.1
 * @date    2020-12-31
 */

#include <gtest/gtest.h>
#include "../Neuron.hpp"

using namespace NN;

TEST ( Neuron_Tests, Empty_Initialization )
{
    Neuron* n = new Neuron ();

    EXPECT_EQ ( n->getBias (), 0 );
    EXPECT_EQ ( n->getNumWeights (), 0 );

    delete n;
}

TEST ( Neuron_Tests, Set_Weights )
{
    float weights [] = { 0.1f, 0.5f, -.2f, .3f, -.7f };
    Neuron* n = new Neuron ( 5 );
    n->setWeights ( weights );

    for ( int i = 0; i < 5; i ++ ) 
    {
        EXPECT_EQ ( n->getWeight ( i ), weights [i] );
    }

    delete n;
}

TEST ( Neuron_Tests, Resize_Weights )
{
    float weights [] = { 0.1f, 0.5f, -.2f, .3f, -.7f };
    Neuron* n = new Neuron ();

    n->resize ( 5 );
    n->setWeights ( weights );
    EXPECT_EQ ( n->getNumWeights (), 5 );
    for ( int i = 0; i < 5; i ++ )
    {
        EXPECT_EQ ( n->getWeight (i), weights [i] );
    }

    n->resize ( 3 );
    EXPECT_EQ ( n->getNumWeights (), 3 );
    for ( int i = 0; i < 3; i ++ )
    {
        EXPECT_EQ ( n->getWeight (i), weights [i] );
    }
    
    delete n; 
}

TEST ( Neuron_Tests, Set_Bias )
{
    Neuron n = Neuron ();
    n.setBias ( -3.5f );
    EXPECT_EQ ( n.getBias (), -3.5f );
}

TEST ( Neuron_Tests, Set_Activation )
{
    Neuron n = Neuron ();
    n.setActivation ( -3.5f );
    EXPECT_EQ ( n.getActivation (), -3.5f );
}
