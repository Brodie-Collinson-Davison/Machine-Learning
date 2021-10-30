using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace NN
{
    [Serializable]
    public class NeuralNetwork
    {
        public int NUM_INPUTS { get; private set; }

        public Matrix Inputs { get; private set; }
        public Matrix Outputs { get; private set; }

        public List<DenseLayer> Layers;

        public NeuralNetwork ()
        {
            NUM_INPUTS = 0;
            Inputs = new Matrix();
            Outputs = new Matrix();
            Layers = new List<DenseLayer>();
        }

        /// <summary>
        /// Creates a neural network using an argument string to specify the network structure
        /// </summary>
        /// <param name="argString"></param>
        // example args "10 50r 10s 10s 3
        // 10 inputs 2 hidden lauys of size 50 10 3 output, hidden layer activations written as a suffix
        public NeuralNetwork (string argString)
        {
            Inputs = new Matrix();
            Outputs = new Matrix();

            // initialise layers
            Layers = new List<DenseLayer> ();
            
            // split into individual arguments
            string[] args = argString.Split(' ');

            // set inputs 
            int numInputs;
            if (int.TryParse(args[0], out numInputs))
            {
                NUM_INPUTS = numInputs;
            }
            else // invalid input argument
                throw new ArgumentException("input argument invalid for NeuralNetwork constructor");

            // start at i = 1 to skip input layer
            for ( int i = 1; i < args.Length; i ++ )
            {
                // get current argumnet
                string arg = args[i];

                int layerSize;
                int prevLayerSize = (i == 1) ? NUM_INPUTS : Layers[i - 2].Size;
                
                ActivationFunctions func;

                // check if argument is only a layer size
                if (int.TryParse (arg, out layerSize))
                {
                    // add new layer with randomised values
                    Layers.Add(new DenseLayer(layerSize, prevLayerSize));
                }
                else //layer activation function given, need to parse it
                {
                    // parse activation function character
                    func = ParseActivationFunctionFromArgString (arg);

                    // remove activaiton function suffix and parse size
                    string sizeArg = arg.Substring(0, arg.Length - 1);
                    
                    if ( int.TryParse (sizeArg, out layerSize) )
                    {
                        // add new layer with randomised values
                        Layers.Add(new DenseLayer(layerSize, prevLayerSize, func));
                    }
                    else // argument string was given in an invalid format
                    {
                        throw new ArgumentException("Layer argument invalid for NeuralNetwork constructor, ensure all your layer arguments are in the correct format.");
                    }
                }
            }
        }

        /// <summary>
        /// Propagates the input through the network and calculates the output activation
        /// </summary>
        /// <param name="input"></param>
        /// <returns>output activation</returns>
        public Matrix Predict (Matrix input)
        {
            if (input.Rows != NUM_INPUTS || input.Cols != 1)
                throw new ArgumentException("Invalid input given to NeuralNetwork, ensure you are using a correct input");
            else // input is in correct format
            {
                // set input
                Inputs = input;

                Matrix curOutput = Layers[0].FeedForward(Inputs);

                for ( int i = 1; i < Layers.Count; i ++ )
                {
                    curOutput = Layers[i].FeedForward(curOutput);
                }

                Outputs = curOutput;
            }


            return Outputs;
        }    

        public override string ToString()
        {
            string str = "NeuralNetwork\n".PadLeft(30);

            str += $"Inputs: {NUM_INPUTS}\n";

            foreach (DenseLayer l in Layers)
            {
                str += $"{l.ActivationFunction} Layer:\n".PadLeft(25);
                str += "Weights:\n";
                str += l.Weights.ToString() + "\n";
                str += "Biases:\n";
                str += l.Biases.ToString() + "\n";
            }

            return str;
        }

        // parses an argument from the arguments string and returns its corresponding ActivationFunctions type
        private ActivationFunctions ParseActivationFunctionFromArgString (string arg)
        {
            ActivationFunctions func;

            char c = arg[arg.Length - 1];

            // parse activation suffix char
            switch (c)
            {
                // sigmoid 
                case 's':
                    func = ActivationFunctions.Sigmoid;
                    break;
                
                // relu
                case 'r':
                case 'R':
                    func = ActivationFunctions.Relu;
                    break;

                // unrecoginised activation suffix
                default:
                    throw new ArgumentException("Activaiton function suffix invalid. Check the args string is in the correctr format");
            }

            return func;
        }

    }//NeuralNetwork ()

}//NN{}