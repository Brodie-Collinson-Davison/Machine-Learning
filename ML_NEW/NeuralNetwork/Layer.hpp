/**
 * @file    Layer.hpp
 * @author  Brodie Collinson Davison
 * @brief   Handles representing an array of neurons with matricies
 * @version 0.1
 * @date    2021-01-01
 */

#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

#include "Matrix.hpp"
#include "Neuron.hpp"

using namespace MATRIX;

namespace NN
{
    class Layer {
    private:    //  MEMBERS
        vector <Neuron> neurons;

    public:     //  PUBLIC INTERFACE
        
        //          CONSTRUCTORS            //
        //  default ctor
        Layer ();
        // copy ctor
        Layer ( const Layer& other );
        // create layer with preset weights
        Layer ( const int& numNeurons, 
                const int& numWeights, 
                const bool& randomise );

        //dtor
        ~Layer ();

        //          ACCESSORS           //
        Neuron getNeuron ( const int& idx );

        int getNumNeurons () const;
        int getNumWeights () const;

        //          MUTATORS            //
        void resizeNeurons ( const int& numNeurons );
        void resizeWeights ( const int& numWeights ); 
    };
}


#endif