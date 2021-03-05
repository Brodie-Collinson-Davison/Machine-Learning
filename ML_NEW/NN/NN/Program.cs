
using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Text.Json;
using NN;

// needed for shuffling IEnumerable<T>
public static class EnumerableExtensions
{
    public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> source)
    {
        return source.Shuffle(new Random());
    }

    public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> source, Random rng)
    {
        if (source == null) throw new ArgumentNullException("source");
        if (rng == null) throw new ArgumentNullException("rng");

        return source.ShuffleIterator(rng);
    }

    private static IEnumerable<T> ShuffleIterator<T>(this IEnumerable<T> source, Random rng)
    {
        var buffer = source.ToList();
        for (int i = 0; i < buffer.Count; i++)
        {
            int j = rng.Next(i, buffer.Count);
            yield return buffer[j];

            buffer[j] = buffer[i];
        }
    }
}

class MainClass
{
    static void Main(string[] args)
    {
        NeuralNetwork net = new NeuralNetwork("784 16 16 10", true);
        float accuracy_untrained = testNetwork(net);
        float accuracy_trained = 0;

        Console.WriteLine("Log training (y/n)?");
        string input = Console.ReadLine();
        bool shouldLog = false;
        if (input == "y")
            shouldLog = true;

        Console.Clear();
    
        trainNetwork(net, 0.01f, 150, 0.15f, shouldLog);
        Console.WriteLine("Training done!");

        accuracy_trained = testNetwork(net);
        Console.WriteLine("Untrained: {0}\nTrained: {1}", accuracy_untrained, accuracy_trained);

        Console.WriteLine("Save network as: ");
        input = Console.ReadLine();

        Console.WriteLine("saving as {0}", input);
        FileManager.SerializeAsJson(net, input);
    }

    /// <summary>
    /// Creates an input matrix with normalised values from an MNIST image
    /// and generates the corresponding expected output matrix
    /// </summary>
    /// <param name="img"></param>
    /// <param name="input"> input matrix to adjust </param>
    /// <param name="target"> target matrix to adjust </param>
    static void processImg ( Image img, out Matrix input, out Matrix target )
    {
        float[] vals = new float[28 * 28];
        
        // normalise input data
        for ( int i = 0; i < 28; i ++ )
        {
            for ( int  j = 0; j < 28; j ++ )
            {
                vals[i * 28 + j] = img.data[i, j] / 255f;
            }
        }

        input = new Matrix(28*28, 1, vals);
        target = new Matrix(10, 1);
        target.SetValue(img.label, 0, 1.0f);
    }

    static void trainNetwork ( NeuralNetwork net, float learnRate, int batchSize, float targetCost, bool logTraining = false )
    {
        // read training data
        var data = MNISTReader.ReadTrainingData();

        bool training = true;

        float costSum = 0f;
        float elapsedTime = 0;

        int count = 1;
        int batchCount = 0;
        int epochCount = 0;
        int correct = 0;
        int correctThisBatch = 0;

        Stopwatch watch = new Stopwatch();

        // initialise bias and weight delta storage
        Matrix[] weightChanges = new Matrix[net.getNumLayers()];
        Matrix[] biasChanges = new Matrix[net.getNumLayers()];
        for ( int i = 0; i < net.getNumLayers(); i ++ )
        {
            Layer l = net.Layers[i];
            weightChanges[i] = new Matrix(l.WM.Rows, l.WM.Cols);
            biasChanges[i] = new Matrix(l.BM.Rows, 1);
        }

        // train network
        while ( training ) // epoch loop
        {
            // display epoch number
            Console.SetCursorPosition(0, 0);
            Console.WriteLine("Epoch: " + epochCount);

            // shuffle data at each epoch
            data = data.Shuffle();

            // loop through training data
            foreach (var img in data) 
            {
                // prepare training example
                Matrix input = null;
                Matrix output = null;
                processImg(img, out input, out output);

                // calculate cost and deltas
                Matrix cost = null;
                Matrix[] exampleActivations = null;
                Matrix[] exampleDeltas = net.backProp(input, output, out exampleActivations, out cost);

                // cost
                float sum = 0;
                for (int i = 0; i < 10; i++)
                {
                    sum += cost.GetValue(i, 0);
                }
                costSum += sum;

                // accumulate weighta and bias changes
                for (int i = net.getNumLayers() - 1; i >= 0; i--)
                {
                    Matrix dW = exampleDeltas[i] * exampleActivations[i].Transpose();
                    Matrix dB = exampleDeltas[i];

                    weightChanges[i] += dW;
                    biasChanges[i] += dB;
                }

                // check prediction
                float highestOutput = 0.0f;
                int prediction_int = 0;
                Matrix prediction = net.predict(input);
                for (int i = 0; i < prediction.Rows; i++)
                {
                    if (prediction.GetValue(i, 0) > highestOutput)
                    {
                        highestOutput = prediction.GetValue(i, 0);
                        prediction_int = i;
                    }
                }

                if (prediction_int == img.label)
                {
                    correct++;
                    correctThisBatch++;
                }

                // completed batch 
                if (count % batchSize == 0)
                { 
                    long batchTime = watch.ElapsedMilliseconds;
                    elapsedTime += (float)batchTime / 1000f;

                    // move cursor under epoch util
                    Console.SetCursorPosition(0, 1);
                    
                    // Training info
                    Console.WriteLine($"Batch: {batchCount}");
                    Console.WriteLine("Time elapsed: {0:0.00}\tTime per batch: {1:000}", elapsedTime, batchTime);

                    float avgCost = (float)costSum / (float)(count);
                    Console.WriteLine("\tCost\n" + "Epoch: {0:0.0000}", avgCost);

                    float batchAccuracy = 100.0f * ((correctThisBatch) / (float)batchSize);
                    float totalAccuracy = 100.0f * ((float)correct / (float)count);
                    Console.WriteLine("\tAccuracy\nEpoch: {0:0.00}%\tBatch: {1:0.00}%", totalAccuracy, batchAccuracy);

                    // update weights and biases
                    for (int i = 0; i < net.getNumLayers(); i++)
                    {
                        Layer l = net.Layers[i];

                        Matrix updW = l.WM - learnRate * weightChanges[i];
                        Matrix updB = l.BM - learnRate * biasChanges[i];

                        net.Layers[i].WM = updW;
                        net.Layers[i].BM = updB;

                        // clear changes for next batch
                        weightChanges[i] = 0.0f * weightChanges[i];
                        biasChanges[i] = 0.0f * biasChanges[i];
                    }

                    // check if done training
                    if (avgCost <= targetCost)
                        training = false;

                    // log
                    if (logTraining)
                        LogTrainingData(batchCount, avgCost, totalAccuracy);

                    // reset batch Metrics
                    correctThisBatch = 0;
                    batchCount++;

                    watch.Restart();
                }//if()

                count++;
            }//foreach()

            // finish epoch
            epochCount++;
            count = 1;
            correct = 0;
            costSum = 0;
        }//while()
    }//trainNetwork()

    static float testNetwork ( NeuralNetwork net )
    {
        var images = MNISTReader.ReadTestData();

        float accuracy = 0.0f;
        int count = 1;
        int correct = 0;

        int upd_freq = 1000;

        // test all images
        foreach ( var img in images )
        {
            // prep data
            Matrix input = null;
            Matrix target = null;
            processImg(img, out input, out target);

            // get prediction
            Matrix prediction = net.predict(input);
            float highestOutput = 0.0f;
            int prediction_int = 0;

            for ( int i = 0; i < prediction.Rows; i ++ )
            {
                if ( prediction.GetValue (i, 0) > highestOutput )
                {
                    highestOutput = prediction.GetValue(i, 0);
                    prediction_int = i;
                }
            }

            if (prediction_int == img.label)
                correct++;

            if (count%upd_freq == 0)
            {
                Console.WriteLine("{0} / {1}\t|\t{2}", correct, count, ((float)correct / (float)count) * 100);
            }

            count++;
        }

        // calculate avg
        accuracy = (float)correct / (float)count;
        accuracy *= 100;

        return accuracy;
    }

    //METHODS
    static public void VisualiseData()
    {
        int maxNum = 10;
        int counter = 1;

        foreach (var Image in MNISTReader.ReadTrainingData())
        {
            float[] imgInputs = new float[28 * 28];
            float[] imgExpectedOutput = new float[10];

            for (int i = 0; i < 10; i++)
            {
                if (i == Image.label)
                    imgExpectedOutput[i] = 1;
                else
                    imgExpectedOutput[i] = 0;
            }

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    imgInputs[j + (j * i)] = (float)(Image.data[i, j]);
                }
            }

            Matrix expectedOutput = new Matrix(10, 1, imgExpectedOutput);

            //display img
            string imgStr = "";
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    if ((i == 0 || i == 27) || j == 0)
                        imgStr += "|";

                    char outputChar = 'a';

                    float val = (float)Image.data[i, j] / 255f;

                    if (val < 0.25f)
                        outputChar = '.';
                    else if (val < .5f)
                        outputChar = '*';
                    else
                        outputChar = '#';

                    if (val == 0f)
                        outputChar = ' ';

                    imgStr += outputChar;

                    if (j == 27)
                        imgStr += "|\n";
                }

            }

            Console.WriteLine(imgStr);
            Console.WriteLine(Image.label);
            Console.WriteLine(expectedOutput.ToString());

            if (counter == maxNum)
                break;

            counter++;
        }
    }

    static void LogTrainingData (int step, float cost, float accuracy )
    {
        const string logFileName = "Training_log.csv";
        string data = string.Format($"{step},{cost},{accuracy}");
        FileManager.AppendToFile(logFileName, data);
    }
}