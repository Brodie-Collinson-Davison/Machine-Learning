using System;

namespace NN
{ 
    public enum ActivationFunctions {
        Relu,
        Sigmoid
    }



    public class DenseLayer
    {
        public int Size { get; set; }
        public ActivationFunctions ActivationFunction { get; set; }

        public Matrix Weights { get; set; }
        public Matrix Biases { get; set; }

        public Matrix Z { get; private set; }

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

            // normalise values with the sqrt (Size) to avoid large outputs for big layers 
            float norm = (float)Math.Sqrt(Size);
            for (int i = 0; i < Weights.values.Length; i ++)
            {
                Weights.values[i] /= norm;
            }
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
            // d(l) = ( w(l+1)T * d(l+1) ) .* f_activation_derivative ( z )
            // w(l+1)T * d(l+1) = dCdA(l)
            // d(l) = (dCdA(l)) .* f_activation_derivative (z)
            return (costDerivative).Dot (ApplyActivationFunction (true));
        }

        //
        //          PRIVATE INTERFACE
        //


        /// <summary>
        /// Applies the correct mathematical operation to the weighted sum 'z'
        /// </summary>
        /// <returns></returns>
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
            } else
                throw new InvalidOperationException($"Layer must have a valid ActivationFunctions type. type {ActivationFunction} is invalid");

            return result;
        }

        private void RandomiseWeights ()
        {
            const float MEAN = 0;
            const float STD_DEV = 1f;

            // iterate through all values
            for ( int i = 0; i < Weights.Rows; i ++ )
            {
                for ( int j = 0; j < Weights.Cols; j ++ )
                { 
                    // set value to random normally distributed number
                    Weights.SetValue(i, j, Mathf.rng_normalFloat (MEAN, STD_DEV));
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
