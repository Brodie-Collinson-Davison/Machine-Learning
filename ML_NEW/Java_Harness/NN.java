import java.util.*;
import org.json.*;

public class NN 
{
    int size;
    public Matrix[] weights;
    public Matrix[] biases;
    
    public static void main (String[] args )
    {
        //NN test
        NN net = new NN (FileIO.readJSONFileAsObject
        ("G:/Projects/ML/Machine-Learning/ML_NEW/Java_Harness/network_trained.json"));

        int[] labels = Formatter.getLabels ();
        List<int[][]> images = Formatter.getImages ();
        
        Matrix input = Formatter.formatInput (images.get(0));
        Matrix prediciton = net.predict (input);
    }

    public NN (JSONObject obj)
    {
        JSONArray layers = obj.getJSONArray("Layers");
        size = layers.length();
        weights = new Matrix[size];
        biases = new Matrix[size];

        for ( int i = 0; i < layers.length(); i ++ )
        {
            JSONObject layer = layers.getJSONObject(i);
            JSONObject wm = layer.getJSONObject("WM");
            JSONObject bm = layer.getJSONObject("BM");

            weights[i] = new Matrix (wm);
            biases[i] = new Matrix (bm);
        }
    }

    public Matrix predict (Matrix input)
    {
        Matrix prevActivation = input;

        for ( int i = 0; i < size; i ++ )
        {
            Matrix ws = weights[i].matMul (prevActivation);
            ws = ws.matAdd (biases[i]);

            Matrix activation = Matrix.sigmoid (ws);
            prevActivation = activation;
        }

        return prevActivation;
    }

    public void print ()
    {
        for ( int i = 0; i < size; i ++ )
        {
            System.out.printf ("Layer %d", i);
            
            System.out.println ("Weights");
            weights[i].print ();
            System.out.println ("Biases");
            biases[i].print ();
        }
    }
}
