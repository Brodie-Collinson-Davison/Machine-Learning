/**
 * @file Neuron.cpp
 * @author Brodie Collinson Davison
 * @brief Neuron class represents a neuron in a neural network.
 * @version 0.1
 * @date 2020-11-09\
 */

#include "Neuron.hpp"

using namespace NN;

//              CONSTRUCTORS                //

// allocate weights 
Neuron::Neuron ( int numWeights )
{
    this->bias = 0;
    this->weights = new float [numWeights] ;
}

Neuron::~Neuron ()      //      DESTRUCTOR
{
    if ( weights != nullptr )
    {
        delete[] weights;
    }
}

//              ACCESSORS               //

int Neuron::getNumWeights () const 
{
    return sizeof ( this->weights ) / sizeof ( float );
}

/**
 *@brief Return the idx'th weight in the from float* weights 
 */
float Neuron::getWeight ( int idx ) const
{
    if ( idx < 0 || idx > getNumWeights () )
    {
        throw "Weight Index in Neuron out of bounds!";
    }
    else
    {
        return this->weights[idx];
    }
}

float* Neuron::getWeights () const
{
    return this->weights; 
}

float Neuron::getBias () const
{
    return this->bias;
}

//              MUTATORS                //

void Neuron::setBias ( const float& bias )
{
    this->bias = bias;
} 

void Neuron::setWeights ( float* weights )
{
    this->weights = weights;
}

void Neuron::setWeight ( const int& idx, const float& value )
{
    if ( idx < getNumWeights () )
    {
        this->weights [idx] = value; 
    }
}