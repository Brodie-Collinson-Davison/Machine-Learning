/**
 * @file    Layer.cpp
 * @author  Brodie Collinson Davison
 * @brief   Handles representing an array of neurons with matricies
 * @version 0.1
 * @date    2021-01-01
 */

#include "Layer.hpp"

using namespace NN;

//==================================//
//          CONSTRUCTORS            //
//==================================//

Layer::Layer ()
: neurons ( 0 )
{ }

Layer::Layer ( const Layer& other )
:   neurons ( other.getNumNeurons (), other.getNumWeights () )
{ }

/**
 * @brief Alt CTOR
 *        Creates a layer with defined number of neurons and weights
 * @param numNeurons 
 * @param numWeights 
 * @param randomise  - generate random weights 
 */
Layer::Layer (  const int& numNeurons, 
                const int& numWeights, 
                const bool& randomise )
{
    neurons = vector<Neuron> ( numNeurons, numWeights );
}

//dtor
Layer::~Layer ()
{
    neurons.clear ();
}

//==================================//
//          ACCESSORS               //
//==================================//

Neuron Layer::getNeuron ( const int& idx )
{
    Neuron n = NULL;

    if ( !this->neurons.empty () && ! idx < 0 && idx < getNumNeurons () )
    {
        n = this->neurons [idx];
    }

    return n;
}

/**
 * @brief number of neurons in the layer
 * @return int 
 */
int Layer::getNumNeurons () const
{
    return this->neurons.size ();
}

/**
 * @brief number of weights shared by all neurons in the layer
 * @return int 
 */
int Layer::getNumWeights () const
{
    int numWeights = 0;

    // check if empty
    if ( !this->neurons.empty () )
    {
        numWeights = this->neurons [0].getNumWeights ();
    }

    return numWeights;
}

//==================================//
//          MUTATORS                //
//==================================//

/**
 * @brief Change the number of neurons held in the layer
 * will add to neuron vector if new size is greater
 * will remove neurons from the tail of the vector if the new size is smaller 
 * @param numNeurons - new number of neurons to contain 
 */
void Layer::resizeNeurons ( const int& numNeurons )
{
    int numWeights = 0;

    if ( !this->neurons.empty () )
    {
        numWeights = this->neurons [0].getNumWeights ();
    }

    if ( this->getNumNeurons () < numNeurons )
    {
        // emplace new Neurons 
        for ( int i = 0; i < (numNeurons - this->getNumNeurons ()); i ++ )
        {
            Neuron n = Neuron ( numWeights );
            this->neurons.push_back ( n );
        }
    }
    else
    {
        // remove excess neurons
        for ( int i = 0; i < (this->getNumNeurons () - numNeurons); i ++ )
        {
            neurons.pop_back ();
        }
    }
    
}

/**
 * @brief Change the number of weights held by neurons in the layer
 * @param numWeights 
 */
void Layer::resizeWeights ( const int& numWeights )
{
    //TODO
}