
import java.util.List;

public class Formatter {
    
    private static final String MNIST_TEST_LABELS_PATH = "G:/Projects/Machine-Learning/ML_NEW/Java_Harness/src/MNIST_Visualiser/train-labels.idx1-ubyte";
    private static final String MNIST_TEST_IMAGES_PATH = "G:/Projects/Machine-Learning/ML_NEW/Java_Harness/src/MNIST_Visualiser/train-images.idx3-ubyte";
    
    public static void main ( String[] args )
    {
        int[] labels = getLabels ();
        List<int[][]> images = getImages ();

        System.out.println ( labels [0] );
    }

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
}
