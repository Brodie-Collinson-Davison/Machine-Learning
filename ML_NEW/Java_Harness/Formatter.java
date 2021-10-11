
import java.util.List;

public class Formatter {
    
    private static final String MNIST_LABELS_PATH = "/MNIST_Dataset/train-labels.idx1-ubyte";
    private static final String MNIST_IMAGES_PATH = "/MNIST_Dataset/train-images.idx3-ubyte";

    public static int[] getLabels (String pathToCurDir) throws RuntimeException
    {
        int [] labels = null; 
        
        try
        {
            labels = MnistReader.getLabels ( pathToCurDir.concat(MNIST_LABELS_PATH) );
        }
        catch ( Exception e )
        {
            throw new RuntimeException ( e );
        }

        return labels; 
    }

    public static List<int[][]> getImages (String pathToCurDir)
    {
        List<int[][]> images = null;

        try
        {
            images = MnistReader.getImages ( pathToCurDir.concat(MNIST_IMAGES_PATH) );
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
