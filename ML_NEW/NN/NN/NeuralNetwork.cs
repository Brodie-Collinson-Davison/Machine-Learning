using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace NN
{
    [Serializable]
    public class NeuralNetwork 
    {
        [JsonInclude]
        public Matrix Inputs { get; set; }
        [JsonInclude]
        public Layer[] Layers { get; set; }

        //
        //  ****    CONSTRUCTORS    ****
        //

        // default constructor
        public NeuralNetwork ()
        {
            Layers = new Layer[0];
            Inputs = new Matrix();
        }

        // copy ctor
        public NeuralNetwork ( NeuralNetwork other )
        {
            // copy layers
            this.Layers = new Layer[other.getNumLayers()];
            for (int i = 0; i < other.getNumLayers(); i++)
            {
                this.Layers[i] = new Layer(other.Layers[i]);
            }
            Inputs = new Matrix (other.Inputs);
        }

        /// <summary>
        /// Creates a new network with layers given in the args string
        /// E.G "3 5 2" will make a network with 3 inputs 1 hidden layer with 5 neurons and 2 output neurons 
        /// </summary>
        /// <param name="args">Space separated integers representing layer sizes</param>
        /// <param name="randomize"> if true, weights will be randomized for the network </param>
        public NeuralNetwork(string args, bool randomize = false)
        {
            // best practice is to keep weights centered around 0 and not small
            const float randomization_std_dev = 0.85f;
            const float randomization_mean = 0;

            int[] layerSizes;

            // parse args string
            try
            {
                string[] tokens = args.Split(" ");
                layerSizes = new int[tokens.Length];

                for (int i = 0; i < tokens.Length; i++)
                {
                    layerSizes[i] = int.Parse(tokens[i]);
                }

                // init inputs
                Inputs = new Matrix(layerSizes[0], 1);

                // init hidden & output layer
                Layers = new Layer[layerSizes.Length - 1];
                for (int i = 1; i < layerSizes.Length; i++)
                {
                    Layers[i - 1] = new Layer(layerSizes[i], layerSizes[i - 1]);
                }

                // gen random weights
                if (randomize)
                {
                    // go through each layer's weight matrix
                    for (int idx = 0; idx < Layers.Length; idx++)
                    {
                        Matrix mat = Layers[idx].WM;

                        for (int i = 0; i < mat.Rows; i++)
                        {
                            for (int j = 0; j < mat.Cols; j++)
                            {
                                // weights are normally distributed 
                                float val = Mathf.rng_normalFloat(randomization_mean, randomization_std_dev);

                                // weight magnitudes are normalized by the sqrt(layer size) to avoid large outputs
                                val /= (float)Math.Sqrt(mat.Rows);
                                mat.SetValue(i, j, val);
                            }
                        }

                        // update weight matrix with random values
                        Layers[idx].WM = mat;
                    }
                }

            }
            catch (Exception e)
            {
                Layers = new Layer[0];
                Inputs = new Matrix();

                Console.WriteLine(e.Message);
            }

        }// ctor(string)

        // accessors

        public bool isEmpty ()
        {
            return getNumLayers() == 0;
        }

        public int getNumLayers ()
        {
            return Layers.Length;
        }

        // NN functions

        /// <summary>
        /// Returns the prediction (output) matrix for the given input by propagating the signal through the network
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix predict ( Matrix input )
        {
            this.Inputs = input;
            Matrix prevActivation = Inputs;

            for ( int i = 0; i < Layers.Length; i ++ )
            {
                Matrix output = Layers[i].activation(prevActivation);
                prevActivation = output;
            }

            return prevActivation;
        }

        /// <summary>
        /// Returns the cost matrix for the given example
        /// Records all layer activations (including the input) and saves them in layerActivations
        /// </summary>
        /// <param name="input"></param>
        /// <param name="target"> Expected value </param>
        /// <param name="layerActivations"> Record of all inputs to layers </param>
        /// <returns></returns>
        public Matrix predict_cost ( Matrix input, Matrix target, out Matrix[] layerActivations )
        {
            //prepare inputs
            this.Inputs = input;
            Matrix activation = input;
            layerActivations = new Matrix[Layers.Length+1];
            layerActivations[0] = input;

            for ( int i = 0; i < Layers.Length; i ++ )
            {
                Matrix output = Layers[i].activation(activation);
                layerActivations[i+1] = output;
                activation = output;
            }

            // C = 1/2 * (Y-A)^2
            Matrix cost = activation - target;
            cost = cost.Pow(2);
            cost = 0.5f * cost;
            return cost;
        }

        /// <summary>
        /// Calculates all layer deltas for the given training example (input, target)
        /// Stores all activations and the example cost in out params
        /// </summary>
        /// <param name="input"></param>
        /// <param name="target"></param>
        /// <param name="layerActivations"></param>
        /// <param name="costMat"></param>
        /// <returns></returns>
        public Matrix[] backProp ( Matrix input, Matrix target, out Matrix[] layerActivations, out Matrix costMat )
        {
            Matrix[] layerDeltas = new Matrix[Layers.Length];

            // calculate cost and its derivative w.r.t activations
            costMat = predict_cost(input, target, out layerActivations);
            // C(t)     = 1/2 * (y - t)^2
            // C'(t)    = y - t
            Matrix delCost = layerActivations[Layers.Length] - target;

            // calculate output layer delta
            // del(L) = (A(L) - Y) * S'(Z(L))
            Matrix prevDelta = delCost.Dot(Mathf.Sigmoid(Layers[Layers.Length - 1].weightedSum(layerActivations[Layers.Length - 1]), true));
            layerDeltas[Layers.Length - 1] = prevDelta;

            // backprop delta through network
            for ( int i = Layers.Length - 2; i >= 0; i -- )
            {
                Matrix delta = Mathf.Sigmoid(Layers[i].weightedSum(layerActivations[i]), true);
                delta = delta.Dot(Layers[i+1].WM.Transpose() * prevDelta);
                layerDeltas[i] = delta;
                prevDelta = delta;
            }

            return layerDeltas;
        }

        public override string ToString()
        {
            String s = "";

            int idx = 0;
            foreach ( Layer l in Layers )
            {
                s += "Layer " + idx + "\n";
                s += "Neurons: " + l.Size + "\n";
                s += l.ToString() + "\n";

                idx++;
            }

            return s;
        }

    }//NeuralNetwork{}

}//NN{}