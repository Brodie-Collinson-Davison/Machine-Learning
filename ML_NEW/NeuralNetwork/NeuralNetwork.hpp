/**
 * @file NeuralNetwork.hpp
 * @author Brodie Collinson Davison
 * @brief Neural network class for creating simple feed forward networks
 * @version 0.1
 * @date 2020-11-08
 */

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <iostream>

#include "Neuron.hpp"
#include "Matrix.hpp"

using namespace std;
using namespace MATRIX;

namespace NN
{
    class NeuralNetwork
    {
    public:
        //          CONSTRUCTORS            //
        NeuralNetwork ();
        NeuralNetwork ( int numLayers, int* layerSizes );

        ~NeuralNetwork ();  // DESTRUCTOR

        //          ACCESSORS           //
        int getNumLayers () const;
        
        int getNumInputs () const;
        int getNumOutputs () const;

        int getNumHiddenLayers () const;
        int* getLayerSizes () const;

        Matrix<float> getWeightMatrix ( const int& layerIdx ) const;

        //          MUTATORS            //


    private:

        int numInputs;
        int numOutputs;

        int numLayers;
        int* layerSizes;

        vector <vector <Neuron>> neurons;

    };//NeuralNetwork{}
}

#endif