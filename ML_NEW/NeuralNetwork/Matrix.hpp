/**
 * @file    Matrix.hpp
 * @author  Brodie Collinson Davison
 * @brief   Lightweight matrix class to handle basic matrix arithmetic 
 * @version 0.1
 * @date    2020-11-21
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <memory>
#include <iostream>
#include <exception>
using namespace std;

namespace MATRIX
{

template <typename T>
class Matrix
{
public:
    Matrix ();

    Matrix ( const Matrix& other );

    Matrix ( int rows, int cols );
    Matrix ( int rows, int cols, T vals[] );
    ~Matrix ();
    
    //          ACCESSORS           //
    bool isEmpty () const;
    bool isFull () const;

    int getRows () const;
    int getCols () const;
    T** getData () const;

    T getValue ( const int& i, const int& j ) const;
    
    //          MUTATORS            //
    void setValue ( const int& i, const int& j, const T& val );

    //          UTIL            //
    void print ();

    Matrix<T> identity ( const int& size );
    Matrix<T> transpose ();

    //          OP Overloads        //

    void operator *= ( T& c );                             // M *= const
    Matrix<T> operator + ( Matrix<T>& mat );                  // M += const
    Matrix<T> operator * ( Matrix<T>& mat );

private:
    int rows;
    int cols;
    
    T **data;
};

//              DEFUALT CONSTRUCTOR
template <typename T>
Matrix<T>::Matrix ()
{
    this->rows = 0;
    this->cols = 0;
    
    // initialise data storage on heap
    data = nullptr;
}

//              Copy constructor
template <typename T>
Matrix<T>::Matrix ( const Matrix& other )
{
    this->rows = other.rows;
    this->cols = other.cols;

    // may have to delete stored data

    this->data = new T* [rows];
    for ( int i = 0; i < this->rows; i ++ )
    {
        this->data [i] = new T [cols];

        for ( int j = 0; j < this->cols; j ++ )
        {
            this->data [i][j] = other.getValue ( i, j );
        }
    }
}

//              Empty constructor
template <typename T>
Matrix<T>::Matrix ( int rows, int cols )
{
    this->rows = rows;
    this->cols = cols;

    // initialise as empty matrix (full of zeros)
    this->data = new T* [rows];
    for ( int i = 0; i < rows; i ++ )
    {
        this->data [i] = new T [cols];

        for ( int j = 0; j < cols; j ++ )
        {
            this->data [i][j] = (T)0;
        }
    }
}

//              Assign with array of values
template <typename T>
Matrix<T>::Matrix ( int rows, int cols, T *vals )
{
    this->rows = rows;
    this->cols = cols;

    // initialise data storage on heap
    this->data = new T* [rows];
    for ( int i = 0; i < rows; i ++ )
    {
        this->data [i] = new T [cols];
    
        // fill values in new columns
        for ( int j = 0; j < cols; j ++ )
        {
            this->data [i][j] = vals [ i*cols + j ];
        }
    }
}

template<typename T>
Matrix<T>::~Matrix ()           //  DESTRUCTOR
{
    if ( data != nullptr )
    {
        for ( int i = 0; i < this->rows; i ++ )
        {
            delete[] this->data [i];
        }

        delete[] data;
    }
}

//==============================================//
//              PUBLIC INTERFACE                //
//==============================================//

//              ACCESSORS               //
template <typename T>
int Matrix<T>::getRows () const
{
    return this->rows;
}

template <typename T>
int Matrix<T>::getCols () const
{
    return this->cols;
}

template <typename T>
T** Matrix<T>::getData () const
{
    return this->data;
}

template <typename T>
T Matrix<T>::getValue ( const int& i, const int& j ) const
{
    T val;

    try
    {
        if ( this->data == nullptr )
        {
            throw  new invalid_argument ( "Matrix doesn't contain any data" );
        }
        else if ( i < 0 || i >= this->rows ||
             j < 0 || j >= this->cols )
        {
            char msg [128];
            sprintf_s ( msg, "Value(%d,%d) is out range for Matrix(%d,%d)", i, j, rows, cols );
            throw new invalid_argument ( msg );
        }
        else
        {
            // valid inputs, return data
            val = this->data[i][j];
        }
    }
    catch ( invalid_argument e )
    {
        cerr << e.what () << endl;
        val = NULL;
    }

    return val;
}

//              MUTATORS                //

template <typename T>
void Matrix<T>::setValue ( const int& i, const int& j, const T& inVal )
{
    try
    {
        if ( this->data == nullptr )
        {
            throw new exception ( "Matrix has no data allocated!" );
        }
        else if ( i < 0 || i >= rows ||
                  j < 0 || j >= cols )
        {
            char msg [128];
            sprintf_s ( msg, "Value(%d,%d) is out range for Matrix(%d,%d)", i, j, rows, cols );
            throw new invalid_argument ( msg );
        }
        else
        {
            // no exceptions, safe to set value
            this->data [i][j] = inVal;
        }
    }
    catch( const std::exception& e)
    {
        std::cerr << e.what() << endl;
    }
}


//              UTIL                //

template<typename T>
void Matrix<T>::print ()
{
    for ( int i = 0; i < rows ; i ++ )
    {
        for ( int j = 0; j < cols; j ++ )
        {
            cout << data[i][j] << "\t";
        }

        cout << endl;
    }
}

template <typename T>
Matrix<T> Matrix<T>::identity ( const int& size )
{
    // initialize 0 matrix 
    Matrix<T> mat ( size, size );

    // set all diagonal elements to 1
    for ( int i = 0; i < size; i ++ )
    {
        mat.setValue ( i, i, 1 );
    }

    return mat;
}

template <typename T>
Matrix<T> Matrix<T>::transpose ()
{
    Matrix<T> mat ( this->cols, this->rows );

    for ( int i = 0; i < this->rows; i ++ )
    {
        for ( int j = 0; j < this->cols; j ++ )
        {
            mat.setValue (j, i, this->getValue (i,j) ); 
        }
    }

    return mat;
}

//==============================================//
//              OPERATOR OVERLOADS              //
//==============================================//

template <typename T>
void Matrix<T>::operator *= ( T& c )
{
    if ( this->data != nullptr )
    {
        for ( int i = 0; i < this->rows; i ++ )
        {
            for ( int j = 0; j < this->cols; j ++ )
            {
                this->data[i][j] *= c;
            }
        }
    }
}

//TODO Implement exception handling
template <typename T>
Matrix<T> Matrix<T>::operator + ( Matrix<T>& mat )
{
    Matrix<T> result ( mat.getRows (), mat.getCols () );

    if ( this->rows == mat.getRows () && 
         this->cols == mat.getCols () )
    {
        for ( int i = 0; i < mat.getRows (); i ++ )
        {
            for ( int j = 0; j < mat.getCols (); j ++ )
            {
                int temp = this->getValue ( i, j );
                temp += mat.getValue ( i, j );
                result.setValue ( i, j, temp );
            }
        }
    }
    else
    {
        cout << "******************************" << endl;
        cerr << "Unable to add matrices:" << endl;
        cout << "******************************" << endl;
        this->print ();
        cout << "******************************" << endl;
        mat.print ();
        cout << "******************************" << endl;
    }
    

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator* ( Matrix<T>& mat )
{
    Matrix<T> result ( this->rows, mat.getCols () );

    if ( this->cols == mat.getRows () )
    {
        for ( int i = 0; i < this->rows; i ++ )
        {
            for ( int j = 0; j < mat.getCols (); j ++ )
            {
                T sum = 0;

                for ( int k = 0; k < this->cols; k ++ )
                {
                    sum += this->getValue (i,k) * mat.getValue (k,j);
                }

                result.setValue ( i, j, sum );
            }
        }
    }
    else
    {
        cout << "******************************" << endl;
        cerr << "Unable to multiply matrices:" << endl;
        cout << "******************************" << endl;
        this->print ();
        cout << "******************************" << endl;
        mat.print ();
        cout << "******************************" << endl;
    }

    return result;
}

}
#endif