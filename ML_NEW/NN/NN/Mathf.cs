using System;

static public class Mathf
{
	static Random rng = new Random(DateTime.Now.Second);

    static float rng_mean = 0.0f;
    static float rng_stdDev = 1.0f;

	static public float Sigmoid(float x, bool derivative = false)
	{
        if (!derivative)
            return (float)(1.0f / (1.0f + Math.Exp(-x)));
        else 
            return (Sigmoid(x) * (1f - Sigmoid(x)));
	}

	static public Matrix Sigmoid(Matrix m, bool derivative = false)
	{
		Matrix mat = m;

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

	static public float RandomFloat(float min, float max)
	{
		return (float)rng.NextDouble () * (max - min) + min;
	}

    static public float rng_normalFloat ()
    {
        double u1 = 1.0 - rng.NextDouble(); //uniform(0,1] random doubles
        double u2 = 1.0 - rng.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                     Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
        double randNormal =
                     rng_mean + rng_stdDev * randStdNormal; //random normal(mean,stdDev^2)
        return (float)randNormal;
    }
}

