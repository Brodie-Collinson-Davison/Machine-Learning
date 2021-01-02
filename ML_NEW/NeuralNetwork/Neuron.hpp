/**
 * @file Neuron.hpp
 * @author Brodie Collinson Davison
 * @brief Neuron class. Represents a neuron in a neural network.
 * @version 0.1
 * @date 2020-11-09
 */

#ifndef NEURON_HPP
#define NEURON_HPP

namespace NN
{
    /**
     * @brief Neuron class represents a node or neuron in a neural network.
     * This class stores the weights and bias values 
     */
    class Neuron
    {
    private:
        //          MEMBERS         //
        int numWeights;

        float activation;
        float bias;
        float* weights;

    public:
        //          CONSTRUCTORS            //
        Neuron ();
        Neuron ( int numWeights );
        Neuron ( const Neuron& other );
        
        ~Neuron ();    // DESTRUCTOR
        
        //          ACCESSORS           //
        int getNumWeights () const;
        float getActivation () const;
        float getBias () const;  
        float getWeight ( int idx ) const;
        float* getWeights () const;
        
        //          MUTATORS            //
        void setBias ( const float& bias );
        void setActivation ( const float& activation );
        void setWeight ( const int& idx, const float& value );
        void setWeights ( float* weights );
        void resize ( const int& numWeights );
    };//Neuron{}

}//NN{}

#endif