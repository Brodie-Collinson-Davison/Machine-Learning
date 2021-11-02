using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Text.Json;

namespace NN
{
    public enum CostFunctions
    {
        SqrError,   // Square Error function
        CCE         // Categorical Cross Entropy
    }

    [Serializable]
    public class NeuralNetwork
    {
        [JsonInclude]
        public CostFunctions CostFunction { get; private set; }
        [JsonInclude]
        public int NUM_INPUTS { get; private set; }
        [JsonInclude]
        public Matrix Inputs { get; private set; }
        [JsonInclude]
        public Matrix Output { get; private set; }
        [JsonInclude]
        public List<DenseLayer> Layers;

        public NeuralNetwork ()
        {
            NUM_INPUTS = 0;
            Inputs = new Matrix();
            Output = new Matrix();
            Layers = new List<DenseLayer>();
            CostFunction = default;
        }

        /// <summary>
        /// Creates a neural network using an argument string to specify the network structure
        /// </summary>
        /// <param name="argString"></param>
        // example args "10 50r 10s 10s 3
        // 10 inputs 2 hidden lauys of size 50 10 3 output, hidden layer activations written as a suffix
        public NeuralNetwork (string argString, CostFunctions costFunc)
        {
            Inputs = new Matrix();
            Output = new Matrix();

            CostFunction = costFunc;

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

            // check if layers have correct form
            foreach (DenseLayer d in Layers)
            {
                if (d.ActivationFunction == ActivationFunctions.SoftMax && (Layers.IndexOf(d) != Layers.Count - 1))
                {
                    // found a softmax layer that isnt at the output layer
                    throw new ArgumentException("Args string contains a softmax layer that isnt the network output");
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

                Output = curOutput;
            }

            return Output;
        }    
        
        public Matrix GetCostDerivative (Matrix expected)
        {
            Matrix del = null;
            switch (CostFunction)
            {
                case CostFunctions.SqrError:
                    del = Output - expected;
                    break;

                case CostFunctions.CCE:
                    del = Output - expected;
                    break;
            }

            return del;
        }

        public float GetCost (Matrix expected)
        {
            float f = 0;

            switch (CostFunction)
            {
                case CostFunctions.SqrError:
                    f = SqrErrorCost(expected);
                    break;

                case CostFunctions.CCE:
                    f = CategoricalCrossEntropyCost(expected);
                    break;
            }

            return f;
        }

        public override string ToString()
        {
            string str = "NeuralNetwork\n".PadLeft(30);

            str += $"Inputs: {NUM_INPUTS}\n";

            foreach (DenseLayer l in Layers)
            {
                str += $"{l.ActivationFunction} [{l.Size}]\t";
                str += "Weights:  ";
                str += l.Weights.ToString();
                str += "\tBiases:  ";
                str += l.Biases.ToString() + "\n";
            }

            return str;
        }

        public override bool Equals(object obj)
        {
            bool equals = true;
            
            // checks if is the same instance
            if (Object.ReferenceEquals(obj, this))
                return true;

            if (obj is NeuralNetwork)
            {
                NeuralNetwork other = (NeuralNetwork)obj;

                if (other.NUM_INPUTS != NUM_INPUTS)
                    equals = false;

                if (other.Layers.Count != Layers.Count)
                    equals = false;
                else
                {
                    for (int i = 0; i < Layers.Count; i++)
                    {
                        if (Layers[i] != other.Layers[i])
                            equals = false;
                    }
                }
            }
            else
                equals = false;

            return equals;
        }

        private float SqrErrorCost(Matrix expected)
        {
            Matrix costmat = 0.5f * (expected - Output).ElementWisePow(2);
            float costSum = 0;

            // sum all output cost elements to get the total cost of the network
            foreach (float f in costmat.values)
            {
                costSum += f;
            }

            return costSum;
        }

        private float CategoricalCrossEntropyCost(Matrix expected)
        {
            float f = 0;

            int maxIdx = Matrix.ArgMaxIndex(expected);

            f = -1f* (float)Math.Log(Output.values[maxIdx]);

            return f / Output.values.Length;
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

                // softmax 
                case 'S':
                    func = ActivationFunctions.SoftMax;
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

        static public NeuralNetwork DeserializeFromJSON (string json)
        {
            NeuralNetwork net = null;

            try
            {
                net = JsonSerializer.Deserialize<NeuralNetwork>(json);
            } catch (Exception e)
            {
                Console.Error.WriteLine (e.Message);
            }

            return net;
        }

        public string GetJSONString ()
        {
            return JsonSerializer.Serialize(this);
        }

    }//NeuralNetwork ()

}//NN{}