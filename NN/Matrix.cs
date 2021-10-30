using System;
using System.Text.Json.Serialization;

[Serializable]
public class Matrix
{
	[JsonInclude]
	public int Rows { get; private set; }
	[JsonInclude]
	public int Cols { get; private set; }
	[JsonInclude]
	public float[] values { get; private set; }

    //
    //	****	CONSTRUCTORS	****
    //

	// default ctor
	public Matrix ()
	{
		Rows = 0;
		Cols = 0;
		values = new float[0];
	}
	
	// copy ctor
    public Matrix ( Matrix other )
    {
        Rows = other.Rows;
        Cols = other.Cols;
        values = other.values;
    }

	/// <summary>
	/// Creates an empty matrix with the specified size 
	/// </summary>
	/// <param name="rows"></param>
	/// <param name="cols"></param>
	public Matrix ( int rows, int cols )
	{
		this.Rows = rows;
		this.Cols = cols;
		this.values = new float [rows*Cols];
	}

	/// <summary>
	/// Constructs a matrix using an args string
	/// e.g "1,2,3;4,5,6;7,8,9" = 3x3 matrix with elements 1 through 9
	/// use ; to denote end of a row and , to separate elements
	/// </summary>
	/// <param name="args"></param>
    public Matrix ( string args )
    {
        string[] rowTokens = args.Split(';');
		string[] nums = rowTokens[0].Split(',');

        this.Rows = rowTokens.Length;
        this.Cols = nums.Length;
        this.values = new float[Rows * Cols];

        int i = 0, j = 0, rowSize = nums.Length;
        foreach (string rowTok in rowTokens)
        {
			// get numbers in row
            nums = rowTok.Split(',');

			// check rows are the same size
			if (nums.Length != rowSize)
				throw new ArgumentException($"The args string: '{args}' is not valid, rows must be the same size!");

			// parse each number in row
			foreach (string num in nums)
            {
                float.TryParse(num, out values[i * Cols + j]);
                j++;
            }

            j = 0;
            i++;
        }
    }

	/// <summary>
	/// Constructor with an array of values
	/// </summary>
	/// <param name="_rows"></param>
	/// <param name="_columns"></param>
	/// <param name="_values"></param>
	public Matrix (int _rows, int _columns, float[] _values)
	{
		// check arguments
		if (_rows <= 0 || _columns <= 0 || _values.Length != _rows * _columns)
        {
			throw new ArgumentException($"Matrix constructor arguments are invalid: R = {_rows} C = {_columns} Data = {_values}");
        }

		Rows = _rows;
		Cols = _columns;
        values = new float[_values.Length];

        // copy values
        for ( int i = 0; i < values.Length; i ++ )
        {
            values[i] = _values[i];
        }
	}

    /// <summary>
    /// Returns the index of the values array for the largest value
    /// </summary>
    /// <returns>index of the largest value in values array</returns>
    public static int ArgMaxIndex (Matrix m)
    {
        float max = float.MinValue;
        int idx = 0;
        int maxIdx = 0;

        foreach (float f in m.values)
        {
            if (f > max)
            {
                max = f;
                maxIdx = idx;
            }

            idx++;
        }

        return maxIdx;
    }

    public static float ArgMax (Matrix m)
    {
        float max = float.MinValue;

        foreach (float f in m.values)
        {
            if (f > max)
                max = f;
        }

        return max;
    }

	public float GetValue(int row, int column) 
	{
		if (row < Rows && column < Cols && row >= 0 && column >= 0)
			return values[row * Cols + column];
		else
        {
			string msg = $"Trying to access element [{row},{column}] in matrix of size [{Rows},{Cols}]";
			throw new ArgumentOutOfRangeException (msg);
        }
	}

	/// <summary>
	/// Creates a square diagonal matrix with ones along the main diagonal
	/// </summary>
	/// <param name="size">size of the matrix</param>
	/// <returns></returns>
	public static Matrix Identity (int size)
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

	/// <summary>
	/// Creates a square diagonal matrix with the value specified
	/// </summary>
	/// <param name="size">size of the matrix</param>
	/// <param name="value">value of each diagonal entry</param>
	/// <returns></returns>
	public static Matrix Diagonal (int size, float value)
	{
		return value * Identity (size);
	}

	/// <summary>
	/// Returns a new matrix with transposed elements
	/// m[i,j] -> m[j,i]
	/// </summary>
	/// <returns></returns>
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

	/// <summary>
	/// Elementwise power operation
	/// </summary>
	/// <param name="p"></param>
	/// <returns>new Matrix with calculated values</returns>
    public Matrix ElementWisePow ( double p )
    {
		Matrix result = new Matrix (this);

		for (int i = 0; i < values.Length; i ++)
		{
			result.values [i] = (float)Math.Pow (values[i], p);
		}

		return result;
    }

    /// <summary>
    /// Element wise square root 
    /// </summary>
    /// <returns>new Matrix with calculated values</returns>
    public Matrix ElementWiseSqrt ()
    {
        return ElementWisePow(0.5);
    }

	/// <summary>
	/// Calculates the dot product of two matrices. 
	/// Matrices must be the same size 
	/// </summary>
	/// <param name="m"></param>
	/// <returns>new matrix equal to this .* m</returns>
	public Matrix Dot(Matrix m)
	{
		Matrix v = new Matrix(this);

		// check dimensions match
		if (this.Rows == m.Rows && this.Cols == m.Cols)
		{
			for (int i = 0; i < this.Rows; i++)
			{
				for (int j = 0; j < this.Cols; j++)
				{
					// multiply all values element-wise and store in v
					v.SetValue(i, j, this.GetValue(i, j) * m.GetValue(i, j));
				}
			}
		}
		else // throws exception
        { 
            throw new ArgumentException($"Matrices must have same dimensions to perform dot operation.\n[{this.Rows},{this.Cols}] .* [{m.Rows}],{m.Cols}]");
		}

		return v;
	}

	//
	//	****	MUTATOR METHODS	   ****
	//

	/// <summary>
	/// Sets the value of the row i, column j element of the matrix to the specified value
	/// </summary>
	/// <param name="row"></param>
	/// <param name="column"></param>
	/// <param name="value"></param>
	public void SetValue(int row, int column, float value)
	{
		// can only set value of an element that exists
		if (row < Rows && column < Cols && row >= 0 && column >= 0)
			values [row * Cols + column] = value;
		else
			throw new ArgumentOutOfRangeException($"Unable to set value {value} at position [{row}, {column}] in Matrix of size [{Rows}, {Cols}");
	}

    //
    //	****	OPERATOR OVERLOADS	****	
    //

	/// <summary>
	/// Matrix addition operation m1 + m2
	/// </summary>
	/// <param name="m1"></param>
	/// <param name="m2"></param>
	/// <returns>New matrix with the values m1 + m2</returns>
	static public Matrix operator + (Matrix m1, Matrix m2)
	{
		// addition of matrices only possible when matrices have same size 
		if (m1.Rows == m2.Rows && m1.Cols == m2.Cols) {
			float[] newValues = new float[m1.values.Length];

			for (int i = 0; i < newValues.Length; i++) {
				newValues [i] = m1.values [i] + m2.values [i];
			}

			return new Matrix (m1.Rows, m1.Cols, newValues);
		}
		else // throw exception
		{
			throw new ArgumentException($"matrix opperation M1 * M2 is invalid!\n[{m1.Rows},{m1.Cols}] x [{m2.Rows},{m2.Cols}]");
		}
	}

	/// <summary>
	/// Matrix subtraction operation m1 - m2
	/// </summary>
	/// <param name="m1"></param>
	/// <param name="m2"></param>
	/// <returns>A new Matrix with values m1 - m2</returns>
    static public Matrix operator - (Matrix m1, Matrix m2)
    {
		// check if dimensions are consistent
        if (m1.Rows == m2.Rows && m1.Cols == m2.Cols)
        {
            float[] newValues = new float[m1.values.Length];

            for (int i = 0; i < newValues.Length; i++)
            {
                newValues[i] = m1.values[i] - m2.values[i];
            }

			// create new matrix
            return new Matrix(m1.Rows, m1.Cols, newValues);
        }
		else // throw exception
		{
			throw new ArgumentException($"matrix opperation M1 * M2 is invalid!\n[{m1.Rows},{m1.Cols}] x [{m2.Rows},{m2.Cols}]");
		}
	}

	/// <summary>
	/// Matrix multiplication operation m1 * m2
	/// </summary>
	/// <param name="m1"></param>
	/// <param name="m2"></param>
	/// <returns>New Matrix with values m1 * m2</returns>
	static public Matrix operator * (Matrix m1, Matrix m2)
	{
		// multiplication only possible if size matches
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
		else // throw exception
        {
			throw new ArgumentException($"matrix opperation M1 * M2 is invalid!\n[{m1.Rows},{m1.Cols}] x [{m2.Rows},{m2.Cols}]");
		}
	}

	/// <summary>
	/// Constant times matrix operator c*M
	/// Multiplies the matrix elementwise
	/// </summary>
	/// <param name="f"></param>
	/// <param name="m"></param>
	/// <returns>New matrix with the values of c * M</returns>
	static public Matrix operator *(float f, Matrix m)
	{
		Matrix result = new Matrix(m);

		for (int i = 0; i < m.values.Length; i++)
		{
			result.values[i] *= f;
		}

		return result;
	}
	
	// equals boolean operator overload
	public static bool operator ==(Matrix lhs, Matrix rhs) => (rhs.Equals(lhs));

	// required for equals operator
	public override int GetHashCode() => (Rows, Cols, values).GetHashCode();
    
	// not equal boolean operator overload
	public static bool operator !=(Matrix lhs, Matrix rhs) => !(lhs == rhs);

	/// <summary>
	/// Override for Object.Equals ()
	/// Checks if two matricies are the same instance or have equal values
	/// </summary>
	/// <param name="obj"></param>
	/// <returns></returns>
	public override bool Equals (Object obj)
    {
		bool result = false;

		// check if is same object
		if (Object.ReferenceEquals(this, obj))
		{
			result = true;
		} 
		else if (obj is Matrix) 
		{
			Matrix other = (Matrix)obj;

			// check if same values
			if (other.values.Length == values.Length)
			{
				for ( int i = 0; i < values.Length; i ++ )
                {
					// failed, exit out of loop 
					if (values[i] != other.values[i])
                    {
						break;
                    }
                }

				result = true;
			}
        }

		return result;
    }

	/// <summary>
	/// Proivdes a string representatoin of the matrix
	/// Overloads Object.ToString()
	/// Will show values for small matrices or the dimensions for large matrices
	/// </summary>
	/// <returns></returns>
	public override string ToString ()
    { 	
        return $"[{Rows}, {Cols}]";
	}

    /// <summary>
    /// Provides a graphical string representation for the matrix 
    /// </summary>
    /// <returns></returns>
	public string GetGraphicalString ()
    {
		string str = "";

		for (int i = 0; i < Rows; i++)
		{
			for (int j = 0; j < Cols; j++)
			{
                string s = String.Format("| {0:.###} ", GetValue(i, j));
                s = s.PadRight(5);
                str += s;
			}
			str += " |\n";
		}
		return str;
	}
}

