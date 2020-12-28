/**
 * @file Matrix_Tests.cpp
 * @author Brodie Collinson Davison
 * @brief Class to test all matrix functions
 * @version 0.1
 * @date 2020-11-18
 */

#include <gtest/gtest.h>
#include "../Matrix.hpp"

using namespace MATRIX;

//==================================================//
//              INITIALIZATION TESTS                //
//==================================================//

TEST ( Matrix_Tests, Empty_Mat_Should_Have_0_Size )
{
    Matrix<int> mat = Matrix<int> ();

    EXPECT_EQ ( mat.getRows (), 0 );
    EXPECT_EQ ( mat.getCols (), 0 );

    mat.~Matrix ();
}

TEST ( Matrix_Tests, Empty_Mat_Should_Have_No_Data )
{
    Matrix<int> mat = Matrix<int> ();
    EXPECT_EQ ( mat.getData (), nullptr );
    
    mat.~Matrix ();
}

TEST ( Matrix_Tests, Array_Initializer_Inserts_Values_Correctly )
{
    int data [] = { 1, 2, 3, 
                    4, 5, 6,
                    7, 8, 9 };

    Matrix<int> mat ( 3, 3, data );

    for ( int i = 0; i < mat.getRows (); i ++ )
    {
        for ( int j = 0; j < mat.getCols (); j ++ )
        {
            EXPECT_EQ ( mat.getValue ( i, j ), data [mat.getCols () * i + j] );
        }
    }
}

TEST ( Matrix_Tests, Identity_Initializer )
{
    Matrix<int> mat = Matrix<int> ().identity ( 3 );

    for ( int i = 0; i < 3; i ++ )
    {
        for ( int j = 0; j < 3; j ++ )
        {
            if ( i == j )
                EXPECT_EQ ( 1, mat.getValue ( i, j ) );
            else
                EXPECT_EQ ( 0, mat.getValue ( i, j ) );
        }
    }
}

TEST ( Matrix_Tests, Transposse_Test )
{
    int data [] =
    {
        1, 2, 3,
        4, 5, 6
    };

    Matrix<int> m ( 2, 3, data );
    Matrix<int> mT = m.transpose ();

    for ( int i = 0; i < 2; i ++ )
    {
        for ( int j = 0; j < 3; j ++ )
        {
            EXPECT_EQ ( m.getValue (i,j), mT.getValue (j,i) );    
        }
    }
}

//==========================================//
//              OPERATOR TESTS              //
//==========================================//

// * const
TEST ( Matrix_Tests, OP_Self_Constant_Multiplication )
{
    float data [] = 
    {   1, 2, 3, 
        4, 5, 6,
        7, 8, 9     
    };

    Matrix<float> mat ( 3, 3, data );
    float constant = 3.0f;

    mat *= constant;

    for ( int i = 0; i < mat.getRows (); i ++ )
    {
        for ( int j = 0; j < mat.getCols (); j ++ )
        {
            EXPECT_EQ ( mat.getValue ( i, j ),
                        constant * data [i*mat.getCols () + j] );
        }
    }
}

// + M
TEST ( Matrix_Tests, OP_Add_Matrix )
{
    int data [] = 
    {   1, 2, 3, 
        4, 5, 6,
        7, 8, 9     
    };

    int rData [] =
    {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };

    Matrix<int> m1 ( 3, 3, data );
    Matrix<int> m2 ( 3, 3, rData );
    Matrix<int> m3 = m1 + m2;

    for ( int i = 0; i < 3; i ++ )
    {
        for ( int j = 0; j < 3; j ++ )
        {   
            // m1 unchanged
            EXPECT_EQ ( m1.getValue ( i, j ), data [i*3 + j] );
            
            // m2 unchanged
            EXPECT_EQ ( m2.getValue ( i, j ), rData [i*3 + j] );

            // m1 + m2 correct
            EXPECT_EQ ( m3.getValue ( i, j ), data [i*3 +j] + rData [i*3 +j] ); 
        }
    }
}

//TODO Implement test to check if adding different size matrices fails
//TODO Implement test to check if multiplying different size matrices fails

TEST ( Matrix_Tests, OP_Multiply_Matrix )
{
    int data [] =
    {
        3, 2, 1,
        1, 2, 3
    };

    int data1 [] =
    {
        2, 2,
        1, 1,
        3, 3
    };

    Matrix<int> m1 ( 2, 3, data );
    Matrix<int> m2 ( 3, 2, data1 );
    Matrix<int> m3 = m1 * m2;

    EXPECT_EQ ( m1.getRows (), m3.getRows () );
    EXPECT_EQ ( m2.getCols (), m3.getCols () );

    EXPECT_EQ ( m3.getValue (0,0), 11 );
    EXPECT_EQ ( m3.getValue (0,1), 11 );
    EXPECT_EQ ( m3.getValue (1,0), 13 );
    EXPECT_EQ ( m3.getValue (1,1), 13 );
}