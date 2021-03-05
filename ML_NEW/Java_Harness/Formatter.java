
import java.util.List;

public class Formatter {
    
    private static final String MNIST_TEST_LABELS_PATH = "G:/Projects/ML/Machine-Learning/ML_NEW/Java_Harness/train-labels.idx1-ubyte";
    private static final String MNIST_TEST_IMAGES_PATH = "G:/Projects/ML/Machine-Learning/ML_NEW/Java_Harness/train-images.idx3-ubyte";

    public static int[] getLabels () throws RuntimeException
    {
        int [] labels = null; 
        
        try
        {
            labels = MnistReader.getLabels ( MNIST_TEST_LABELS_PATH );
        }
        catch ( Exception e )
        {
            throw new RuntimeException ( e );
        }

        return labels; 
    }

    public static List<int[][]> getImages ()
    {
        List<int[][]> images = null;

        try
        {
            images = MnistReader.getImages ( MNIST_TEST_IMAGES_PATH );
        }
        catch ( Exception e )
        {
            throw new RuntimeException ( e );
        }

        return images;
    }

    public static Matrix formatInput (int[][] img)
    {
        Matrix mat = new Matrix (784, 1);

        for ( int i = 0; i < 28; i ++ )
        {
            for ( int j = 0; j < 28; j ++ )
            {
                float norm = (float)img[i][j] / 255.0f;
                mat.values[i*28+j] = norm;
            }
        }

        return mat;
    }
}
