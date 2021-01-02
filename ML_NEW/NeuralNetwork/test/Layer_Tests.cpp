/**
 * @file    Layer_Tests.cpp
 * @author  Brodie Collinson Davison
 * @brief   Tests for Layer class
 * @version 0.1
 * @date    2021-01-01
 */

#include <gtest/gtest.h>
#include <iostream>

#include "../Layer.hpp"

using namespace NN;

TEST ( Layer_Tests, Empty_Initialisation )
{
    Layer* l = new Layer ();

    EXPECT_EQ ( l->getNumWeights (), 0 );
    EXPECT_EQ ( l->getNumNeurons (), 0 );

    delete l;
}

TEST ( Layer_Tests, Alt_CTOR )
{
    Layer* l = new Layer ( 5, 3, false );

    for ( int i = 0; i < l->getNumNeurons (); i ++ )
    {
        EXPECT_EQ ( l->getNeuron ( i ).getNumWeights (), 3 );
    }
    //FIXME allocation may be wrong? delete is throwing an SEH exception
    //delete l;
}

