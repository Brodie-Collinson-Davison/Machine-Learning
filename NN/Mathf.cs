using System;

static public class Mathf
{
    // static random class to avoid constant reseeding
	static Random rng = new Random(DateTime.Now.Second);

	static public float Sigmoid(float x, bool derivative = false)
	{
        if (!derivative)
            return (float)(1.0f / (1.0f + Math.Exp(-x)));
        else 
            return (Sigmoid(x) * (1f - Sigmoid(x)));
	}

    // Sigmoid activation function applied element wise to a matrix
	static public Matrix Sigmoid(Matrix m, bool derivative = false)
	{
        Matrix mat = new Matrix(m);

		for ( int i = 0; i < mat.Rows; i ++ )
        {
            for ( int j = 0; j < mat.Cols; j ++ )
            {
                float f = mat.GetValue(i, j);
                f = Sigmoid(f, derivative);
                mat.SetValue(i, j, f);
            }
        }

		return mat;
	}

    static public Matrix Relu (Matrix m, bool derivative = false)
    {
        const float epsilon = 1e-9f;
        const float clip = 1.0f;
        const float minVal = epsilon;

        Matrix result = new Matrix (m);

        for ( int i = 0; i < result.Rows; i ++ )
        {
            for ( int j = 0; j < result.Cols; j ++ )
            {
                float f = result.GetValue(i, j);

                if (f < 0)
                    f = minVal;
                else if (f > clip)
                    f = clip;
                else if (derivative)
                    f = 1f;
                result.SetValue(i, j, f);
            }
        }

        return result;
    }

    static public Matrix SoftMax (Matrix m)
    {
        Matrix result = new Matrix(m);

        float sum = 0;
        float largestValue = Matrix.ArgMax(m);

        // get sum of all components normalised over the layer 
        for (int i = 0; i < result.values.Length; i ++)
        {
            sum += (float)Math.Exp (result.values [i] - largestValue);
        }

        // calculate individual components
        for ( int i = 0; i < result.values.Length; i ++)
        {
            float f = (float)Math.Exp(result.values[i] - largestValue) / sum;

            result.values[i] = f;
        }

        return result;
    }

    // returns the average of all values in the array
    public static float AverageArray(float[] array)
    {
        float sum = 0;
        foreach (float f in array)
        {
            sum += f;
        }

        return sum / (float)array.Length;
    }

    // returns a uniformly distributed float between min and max
	static public float RandomFloat(float min, float max)
	{
		return (float)rng.NextDouble () * (max - min) + min;
	}

    // Returns normally distributed random number
    static public float rng_normalFloat (float mean, float stdDev)
    {
        double u1 = 1.0 - rng.NextDouble(); //uniform(0,1] random doubles
        double u2 = 1.0 - rng.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                     Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
        double randNormal =
                     mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
        return (float)randNormal;
    }
}

