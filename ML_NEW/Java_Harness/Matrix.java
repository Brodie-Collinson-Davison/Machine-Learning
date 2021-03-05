import org.json.*;

public class Matrix 
{
    public int rows;
    public int cols;
    public float[] values;

    public Matrix ()
    {
        rows = 0;
        cols = 0;
        values = new float[0];
    }

    public Matrix ( int rows, int cols )
	{
		this.rows = rows;
		this.cols = cols;
		this.values = new float [rows*cols];
	}

    public Matrix (int _rows, int _columns, float[] _values)
	{
		rows = _rows;
		cols = _columns;
		values = _values;
	}

    public Matrix (JSONObject obj)
    {        
        rows = obj.getInt ("Rows");
        cols = obj.getInt ("Cols");
        values = new float[rows * cols];

        JSONArray vals = obj.getJSONArray("values");
        
        for ( int i = 0; i < vals.length(); i ++ )
        {
            float val = (float)vals.getDouble(i);
            values[i] = val;
        }
    }

    public void print ()
    {
        for ( int i = 0; i < rows; i ++ )
        {
            String str = "";
            for ( int j = 0; j < cols; j ++ )
            {
                str += values[i*cols +j] + "\t";
            }
            System.out.println (str);
        }
    }

    public float get(int i, int j)
    {
        return values [i*cols + j];
    }

    public static Matrix sigmoid ( Matrix m )
    {
        Matrix mat = m;

        for ( int i = 0; i < mat.values.length; i ++ )
        {
            float f = mat.values[i];
            f = 1.0f / (1.0f + (float)Math.exp (-f));
            mat.values[i] = f;
        }

        return mat;
    }

    public Matrix matAdd ( Matrix mat )
    {
        float[] newVals = values;

        if (cols == mat.cols && rows == mat.rows)
        {
            int idx = 0;
            for ( float f : mat.values )
            {
                newVals[idx] += f;
                idx ++;
            }
        }

        return new Matrix (rows, cols, newVals);
    }

    public Matrix matMul ( Matrix mat )
    {
        float[] vals = new float [rows * mat.cols];
        for ( int i = 0; i < rows; i ++ )
        {
            for ( int j = 0; j < mat.cols; j ++ )
            {
                float sum = 0;
                for ( int k = 0; k < cols; k ++ )
                {
                    sum += get(i, k) * mat.get(k, j);
                }
                vals [i*mat.cols + j] = sum;
            }
        }

        return new Matrix (rows, mat.cols, vals);
    }
}
