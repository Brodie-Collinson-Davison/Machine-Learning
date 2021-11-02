using System;
using System.Text.Json.Serialization;

namespace NN
{ 
    [Serializable]
    public enum ActivationFunctions {
        Relu,
        Sigmoid,
        SoftMax
    }

    [Serializable]
    public class DenseLayer
    {
        [JsonInclude]
        public int Size { get; set; }
        
        [JsonInclude]
        public ActivationFunctions ActivationFunction { get; set; }

        [JsonInclude]
        public Matrix Weights { get; set; }
        [JsonInclude]
        public Matrix Biases { get; set; }
        [JsonInclude]
        public Matrix Z { get; private set; }
        [JsonInclude]
        public Matrix A { get; private set; }


        // empty constructor
        public DenseLayer ()
        {
            Size = 0;
            Weights = new Matrix();
            Biases = new Matrix();
            ActivationFunction = default;
            Z = new Matrix();
            A = new Matrix();
        }

        public DenseLayer (DenseLayer l)
        {
            Size = l.Size;
            Weights = l.Weights;
            Biases = l.Biases;
            ActivationFunction = l.ActivationFunction;
            Z = new Matrix();
            A = new Matrix();
        }

        /// <summary>
        /// Generates layer with random weights and biases
        /// </summary>
        /// <param name="size"></param>
        public DenseLayer (int size, int prevLayerSize, ActivationFunctions activationFunction = default)
        {
            this.Size = size;
            Weights = new Matrix(size, prevLayerSize);
            Biases = new Matrix(size, 1);
            Z = new Matrix();
            A = new Matrix();

            this.ActivationFunction = activationFunction;

            // weight and bias randomisation
            RandomiseWeights();
            //RandomiseBiases();
        }

        /// <summary>
        /// Calculates the output of the layer:
        /// o = f_activation (W*A + B)
        /// </summary>
        /// <param name="input"></param>
        /// <returns>New matrix representing the output of each neuron in the layer</returns>
        public Matrix FeedForward ( Matrix activation )
        {
            // calculate weighted sum z
            Z = Weights * activation + Biases;

            A = ApplyActivationFunction ();

            // returns the output = f_activation (z)
            return A;
        }

        /// <summary>
        /// Calculates the current layer delta using the derivative 
        /// </summary>
        /// <param name="nextLayerDelta"></param>
        /// <param name="nextLayerWeights"></param>
        /// <returns></returns>
        public Matrix BackProp (Matrix costDerivative)
        {
            // softmax layers will only be at the end of the network and the derivative will be calculated there
            if (ActivationFunction == ActivationFunctions.SoftMax)
                return costDerivative;
            else // d(l) = (dCdA(l)) .* f_activation_derivative (z)
                return (costDerivative).Dot(ApplyActivationFunction(true));
        }

        //
        //          PRIVATE INTERFACE
        //


        /// <summary>
        /// Calculates the activation function or its derivative
        /// </summary>
        /// <param name="derivative">when true enables derivative calculation of activation function</param>
        /// <returns>new Matrix</returns>
        public Matrix ApplyActivationFunction( bool derivative = false)
        {
            Matrix result = null;

            if (ActivationFunction == ActivationFunctions.Sigmoid)
            {
                result = Mathf.Sigmoid(Z, derivative);
            }
            else if (ActivationFunction == ActivationFunctions.Relu)
            {
                result = Mathf.Relu(Z, derivative);
            } else if (ActivationFunction == ActivationFunctions.SoftMax)
            {
                // the derivative will never be calculated 
                if (derivative)
                    result = new Matrix (Z);
                else
                    result = Mathf.SoftMax (Z);
            } else
                throw new InvalidOperationException($"Layer must have a valid ActivationFunctions type. type {ActivationFunction} is invalid");

            return result;
        }

        public override bool Equals(object obj)
        {
            bool equals = true;

            if (obj is DenseLayer)
            {
                DenseLayer other = (DenseLayer)obj;

                if (other.Size != Size)
                    equals = false;

                if (Weights != other.Weights)
                    equals = false;

                if (Biases != other.Biases)
                    equals = false;

                // don't include checks of A and Z
            }
            else
                equals = false;

            return equals;
        }

        private void RandomiseWeights ()
        {
            const float MEAN = 0;
            const float STD_DEV = 1.0f;

            // iterate through all values
            for ( int i = 0; i < Weights.Rows; i ++ )
            {
                for ( int j = 0; j < Weights.Cols; j ++ )
                { 
                    // set value to random normally distributed number
                    Weights.SetValue(i, j, Mathf.rng_normalFloat (MEAN, STD_DEV) / (float)Math.Sqrt (Size));
                }
            }
        }

        private void RandomiseBiases ()
        {
            const float MEAN = 0;
            const float STD_DEV = 0.1f;

            for ( int i = 0; i < Biases.Rows; i ++ )
            {
                Biases.SetValue(i, 0, Mathf.rng_normalFloat(MEAN, STD_DEV));
            }
        }
    }
}
