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
// default constructor
Neuron::Neuron ()
{
    this->bias = 0;
    this->activation = 0;
    this->numWeights = 0;
    this->weights = new float [0];
}

// allocate weights 
Neuron::Neuron ( int numWeights )
{
    this->activation = 0;
    this->bias = 0;
    this->numWeights = numWeights;
    this->weights = new float [numWeights];
}

// copy constructor
Neuron::Neuron ( const Neuron& other )
:   weights ( other.weights ), 
    bias ( other.bias ),
    activation ( other.activation ),
    numWeights ( other.numWeights )
{ }

//dtor
Neuron::~Neuron ()      //      DESTRUCTOR
{
    if ( weights != nullptr )
    {
        delete[] weights;
    }
}

//              ACCESSORS               //

float Neuron::getActivation () const
{
    return this->activation;
}

int Neuron::getNumWeights () const 
{
    return this->numWeights;
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

void Neuron::setActivation ( const float& activation )
{
    this->activation = activation;
}

void Neuron::setBias ( const float& bias )
{
    this->bias = bias;
} 

void Neuron::setWeights ( float* weights )
{
    for ( int i = 0; i < numWeights; i ++ )
    {
        this->weights [i] = weights [i];
    }
}

void Neuron::setWeight ( const int& idx, const float& value )
{
    if ( idx < getNumWeights () )
    {
        this->weights [idx] = value; 
    }
}

void Neuron::resize ( const int& numWeights )
{
    float* weights = new float [numWeights];
    float* temp = this->weights;

    int min = (this->numWeights < numWeights)? this->numWeights : numWeights;

    for ( int i = 0; i < min; i ++ )
    {
        weights [i] = this->weights [i];
    }

    this->numWeights = numWeights;
    this->weights = weights;
    delete temp;
}