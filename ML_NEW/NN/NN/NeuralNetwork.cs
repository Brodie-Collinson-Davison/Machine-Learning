using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace NN
{
    [Serializable]
    class NeuralNetwork 
    {
        [JsonIgnore]
        public Matrix Inputs { get; set; }
        [JsonInclude]
        public Layer[] Layers { get; set; }

        // constructors

        public NeuralNetwork ()
        {
            Layers = new Layer[0];
            Inputs = new Matrix();
        }

        /// <summary>
        /// Creates a new network with layers given in the args string
        /// E.G "3 5 2" will make a network with 3 inputs 1 hidden layer of size 5 and 2 
        /// </summary>
        /// <param name="args"></param>
        /// <param name="randomise"> if true, weights will be randomised for the network </param>
        public NeuralNetwork(string args, bool randomise = false)
        {
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
            }
            catch (ArgumentException e)
            {
                layerSizes = null;
                Console.WriteLine(e.Message);
            }

            // init inputs
            Inputs = new Matrix(layerSizes[0], 1);

            // init hidden & output layer
            Layers = new Layer[layerSizes.Length - 1];
            for (int i = 1; i < layerSizes.Length; i++)
            {
                Layers[i-1] = new Layer(layerSizes[i], layerSizes[i - 1]);
            }

            if ( randomise )
            {
                // gen random weights
                // weights are normally distributed 
                // weights are normalised by the sqrt(layer size) to avoid large outputs
                for ( int idx = 0; idx < Layers.Length; idx ++)
                {
                    Matrix mat = Layers[idx].WM;

                    for ( int i = 0; i < mat.Rows; i ++ )
                    {
                        for ( int j = 0; j < mat.Cols; j ++ )
                        { 
                            float val = Mathf.rng_normalFloat();
                            val /= (float)Math.Sqrt(mat.Rows);
                            mat.SetValue(i, j, val);
                        }
                    }

                    Layers[idx].WM = mat;
                }
            }

        }// ctor(string)

        // accessors

        public int getNumLayers ()
        {
            return Layers.Length;
        }

        // NN functions

        /// <summary>
        /// Returns the prediction (output) matrix for the given input
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
        /// <param name="target"></param>
        /// <param name="layerActivations"> Record of all inputs to layers </param>
        /// <returns></returns>
        public Matrix outputCost ( Matrix input, Matrix target, out Matrix[] layerActivations )
        {
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
            cost.ewPow(2);
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
            costMat = outputCost(input, target, out layerActivations);
            Matrix delCost = layerActivations[Layers.Length] - target;

            // calc output layer delta
            // del(L) = (A(L) - Y) o S' (Z(L))
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

    class Layer
    {
	    public int Size { get; set; }
	    public Matrix WM { get; set; }
	    public Matrix BM { get; set; }

        public Layer ()
        {
            Size = 0;
            WM = new Matrix();
            BM = new Matrix();
        }

        public Layer ( int size, int numWeights )
	    {
		    Size = size;
		    WM = new Matrix ( Size, numWeights );
		    BM = new Matrix ( Size, 1 );
	    }

        // layer operations

        /// <summary>
        /// Calculates the weighted sum z = w * input + b
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
	    public Matrix weightedSum ( Matrix input )
	    {
            return WM * input + BM;
	    }

        /// <summary>
        /// Calculate the output activation of the layer a = sigmoid ( w * input + b )
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
	    public Matrix activation ( Matrix input )
	    {
            return Mathf.Sigmoid(WM * input + BM);
	    }
        public override string ToString ()
        {
            string s = "";
            s += "weights:\n";
            s += WM.ToString() +"\n";

            s += "biases:\n";
            s += BM.ToString() +"\n";

            return s;
        }

    }//Layer{}
}//NN{}