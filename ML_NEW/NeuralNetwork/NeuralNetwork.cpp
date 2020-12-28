/**
 * @file    NeuralNetwork.cpp
 * @author  Brodie Collinson Davison
 * @brief   Neural network class functionality
 * @version 0.1
 * @date    2020-11-08
 */

#include "NeuralNetwork.hpp"

using namespace NN;
using namespace std;

//////////////////////////////////////////////
//              CONSTRUCTORS                //
//////////////////////////////////////////////

NeuralNetwork::NeuralNetwork ()
{
    this->numInputs = 0;
    this->numOutputs = 0;
    this->numLayers = 0;
    this->layerSizes = nullptr;
}

NeuralNetwork::NeuralNetwork ( int numLayers, int* layerSizes )
{
    this->numLayers = numLayers;
    this->numInputs = layerSizes [0];
    this->numOutputs = layerSizes [numLayers - 1];
    this->layerSizes = new int [numLayers];

    for ( int i = 0; i < numLayers; i ++ )
    {   
        this->layerSizes [i] = layerSizes [i];
        vector <Neuron> temp;

        for ( int j = 0; j < layerSizes [i]; j ++ )
        {
            Neuron n;

            if ( i > 0 )
                n = Neuron ( layerSizes [i-1] );
            else
                n = Neuron ();
            
            temp.push_back ( n );
        }
        neurons.push_back ( temp );
    }
}

NeuralNetwork::~NeuralNetwork () // Destructor
{
    
}

///////////////////////////////////////
//              ACCESSORS            //
///////////////////////////////////////

int NeuralNetwork::getNumInputs () const
{
    return this->numInputs;
}

int NeuralNetwork::getNumOutputs () const
{
    return this->numOutputs;
}

int NeuralNetwork::getNumLayers () const 
{
    return this->numLayers;
}

int NeuralNetwork::getNumHiddenLayers () const
{
    return this->numLayers - 2;
}

int* NeuralNetwork::getLayerSizes () const
{
    return this->layerSizes;
}

/**
 * @brief
 * Returns the weight matrix of the given layer
 * Weight matrix is constructed as rows consisting of each neurons weights
 * i.e
 * {
 *   { Neuron[0] weight[0], Neuron[0] weight[1], ...},
 *   { Neuron[1] weight[0], ... },
 *  .-
 *  .-
 * };
 * @param layerIdx - index of the layer to get weight matrix of
 */
Matrix<float> NeuralNetwork::getWeightMatrix ( const int& layerIdx ) const
{

    if ( layerIdx < 1 || layerIdx > numLayers - 1 )
    {
        throw "Index must be within network length!";
    }

    //TODO FIXME!
    Matrix<float> mat ( layerSizes [layerIdx], layerSizes [layerIdx - 1] );
    
    for ( int i = 0; i < layerSizes[layerIdx]; i ++ )
    {
        Neuron n = neurons[layerIdx][i];

        for ( int j = 0; j < n.getNumWeights (); j ++ )
        {
            mat.setValue (i, j, n.getWeight ( j ) );
        }
    }

    return mat;
}//getWeightMatrix()

///////////////////////////////////////
//              MUTATORS             //
///////////////////////////////////////

