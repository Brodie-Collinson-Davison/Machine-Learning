using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NN
{
    [Serializable]
    public class Layer
    {
        [JsonInclude]
        public int Size { get; set; }
        [JsonInclude]
        public Matrix WM { get; set; }
        [JsonInclude]
        public Matrix BM { get; set; }

        public Layer()
        {
            Size = 0;
            WM = new Matrix();
            BM = new Matrix();
        }

        public Layer (Layer other)
        {
            this.Size = other.Size;
            this.WM = new Matrix (other.WM);
            this.BM = new Matrix (other.BM);
        }

        public Layer(int size, int numWeights)
        {
            Size = size;
            WM = new Matrix(Size, numWeights);
            BM = new Matrix(Size, 1);
        }

        // layer operations

        /// <summary>
        /// Calculates the weighted sum z = w * input + b
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
	    public Matrix weightedSum(Matrix input)
        {
            return WM * input + BM;
        }

        /// <summary>
        /// Calculate the output activation of the layer a = sigmoid ( w * input + b )
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
	    public Matrix activation(Matrix input)
        {
            return Mathf.Sigmoid(WM * input + BM);
        }
        public override string ToString()
        {
            string s = "";
            s += "weights:\n";
            s += WM.ToString() + "\n";

            s += "biases:\n";
            s += BM.ToString() + "\n";

            return s;
        }

    }//Layer{}
}
