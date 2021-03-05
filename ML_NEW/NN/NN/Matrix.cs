using System;

[Serializable]
public class Matrix
{
	public int Rows { get; set; }
	public int Cols { get; set; }

	public float[] values { get; set; }

    //
    //Constructors
    //

	public Matrix ()
	{
		Rows = 0;
		Cols = 0;
		values = new float[0];
	}
	
    public Matrix ( Matrix other )
    {
        Rows = other.Rows;
        Cols = other.Cols;
        values = other.values;
    }

	public Matrix ( int rows, int cols )
	{
		this.Rows = rows;
		this.Cols = cols;
		this.values = new float [rows*Cols];
	}

    public Matrix ( string args )
    {
        string[] rowTokens = args.Split(";");
        string[] nums = rowTokens[0].Split(",");

        this.Rows = rowTokens.Length;
        this.Cols = nums.Length;
        this.values = new float[Rows * Cols];

        int i = 0, j = 0;
        foreach (string rowTok in rowTokens)
        {
            nums = rowTok.Split(",");
            foreach (string num in nums)
            {
                float.TryParse(num, out values[i * Cols + j]);
                j++;
            }

            j = 0;
            i++;
        }
    }

	public Matrix (int _rows, int _columns, float[] _values)
	{
		Rows = _rows;
		Cols = _columns;
		values = _values;
	}

	public Matrix Identity (int size)
	{
		Matrix mat = new Matrix (size, size, new float[size * size]);

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				
				if (i == j)
					mat.SetValue (i, j, 1);
			}
		}

		return mat;
	}

	public Matrix Diagonal (int size, float value)
	{
		return value * Identity (size);
	}

	public Matrix Diagonal(float[] values)
	{
		Matrix matrix = new Matrix();

		matrix.Rows = matrix.Cols = values.Length;
		matrix.values = new float[values.Length * values.Length];

		for (int i = 0; i < values.Length; i++)
		{
			for (int j = 0; j < values.Length; j++)
			{
				if (i == j)
					matrix.SetValue(i, j, values[i]);
				else
					matrix.SetValue(i, j, 0);
			}
		}

		return matrix;
	}

	public float GetValue(int row, int column)
	{
		if (row < Rows && column < Cols)
			return values [row * Cols + column];

		return 0;
	}

	public void SetValue(int row, int column, float value)
	{
		if (row < Rows && column < Cols)
			values [row * Cols + column] = value;
	}

    public void ewPow ( float p )
    {
        int idx = 0;
        foreach ( float f in values )
        {
            values[idx] = (float)Math.Pow(values[idx], p);
            idx++;
        }
    }

    //
    //  Operations
    //

	public Matrix Dot (Matrix m)
	{
		Matrix v = this;

		if (Rows == m.Rows && Cols == m.Cols) {
			
			for (int i = 0; i < Rows; i++) {
				for (int j = 0; j < Cols; j++) {

					v.SetValue (i, j, GetValue (i, j) * m.GetValue (i, j));
				}
			}

			return v;
		}

        string dimensionDebug = "M1 = " + v.Rows + "x" + v.Cols + " | M2 = " + m.Rows + "x" + m.Cols;
        throw new System.ArgumentException("matrix opperation m1 . m2 is invalid! " + dimensionDebug);
	}

	public Matrix Transpose()
	{
		Matrix v = new Matrix(Cols, Rows, new float[Rows * Cols]);

		for (int i = 0; i < Cols; i++) {
			for (int j = 0; j < Rows; j++) {
				v.SetValue (i, j, GetValue (j, i));
			}
		}

		return v;
	}

    //
    //Operator overloads
    //

	static public Matrix operator + (Matrix m1, Matrix m2)
	{
		if (m1.Rows == m2.Rows && m1.Cols == m2.Cols) {
			float[] newValues = new float[m1.values.Length];

			for (int i = 0; i < newValues.Length; i++) {
				newValues [i] = m1.values [i] + m2.values [i];
			}
			return new Matrix (m1.Rows, m1.Cols, newValues);
		}

        string dimensionDebug = "M1 = " + m1.Rows + "x" + m1.Cols + " | M2 = " + m2.Rows + "x" + m2.Cols;
        throw new System.ArgumentException("matrix opperation m1 + m2 is invalid! " + dimensionDebug);
	}

    static public Matrix operator - (Matrix m1, Matrix m2)
    {
        if (m1.Rows == m2.Rows && m1.Cols == m2.Cols)
        {
            float[] newValues = new float[m1.values.Length];

            for (int i = 0; i < newValues.Length; i++)
            {
                newValues[i] = m1.values[i] - m2.values[i];
            }
            return new Matrix(m1.Rows, m1.Cols, newValues);
        }

        string dimensionDebug = "M1 = " + m1.Rows + "x" + m1.Cols + " | M2 = " + m2.Rows + "x" + m2.Cols;
        throw new System.ArgumentException("matrix opperation m1 - m2 is invalid! " + dimensionDebug);
    }

	static public Matrix operator * (Matrix m1, Matrix m2)
	{
		if (m1.Cols == m2.Rows) {

			float[] newValues = new float[m1.Rows * m2.Cols];

			for (int i = 0; i < m1.Rows; i++) {
				for (int j = 0; j < m2.Cols; j++) {

					float sum = 0;

					for (int k = 0; k < m1.Cols; k++) {

						sum += m1.GetValue (i, k) * m2.GetValue (k, j);
					}

					newValues [i * m2.Cols + j] = sum;
				}
			}
			return new Matrix (m1.Rows, m2.Cols, newValues);
		}

        string dimensionDebug = "M1 = " + m1.Rows + "x" + m1.Cols + " | M2 = " + m2.Rows + "x" + m2.Cols;
        throw new System.ArgumentException("matrix opperation M1 * M2 is invalid! " + dimensionDebug);
	}

	static public Matrix operator * (float f, Matrix m)
	{
		for (int i = 0; i < m.values.Length; i++) {

			m.values [i] *= f;
		}

		return m;
	}

    static public Matrix operator + (float f, Matrix m)
    {
        for (int i = 0; i < m.values.Length; i ++)
        {
            m.values[i] += f;
        }

        return m;
    }

    static public Matrix operator - (float f, Matrix m)
    {
        for (int i = 0; i < m.values.Length; i ++)
        {
            m.values[i] -= f;
        }

        return m;
    }

	public override string ToString ()
	{
		string mat = "";

		for (int i = 0; i < Rows; i++) {
			for (int j = 0; j < Cols; j++) {

				mat += " | " + GetValue (i, j).ToString ();
			}
			mat += " |\n";
		}
		return mat;
	}
}

