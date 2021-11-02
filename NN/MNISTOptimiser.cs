using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace NN
{
    public enum Optimizers
    {
        Batch,
        Momentum,
        RMSProp,
        Adam
    }

    public class TrainingParams
    {

        // default training parameter values
        private const float DEFAULT_ALPHA = 0.01f;
        private const float DEFAULT_BETA1 = 0.9f;
        private const float DEFAULT_BETA2 = 0.999f;
        private const float DEFAULT_TARGET_COST = 0.04f;
        private const int DEFAULT_BATCH_SIZE = 500;
        private const int DEFAULT_MAX_EPOCH = 10;

        // class properties
        public Optimizers Optimizer { get; set; }
        public float Alpha { get; set; }
        public float Beta1 { get; set; }
        public float Beta2 { get; set; }
        public float TargetCost { get; set; }
        public int BatchSize { get; set; }
        public int MaxEpoch { get; set; }

        // constructors

        // default constructor initialises with default values
        public TrainingParams
            (
            Optimizers iOptimizer = Optimizers.Adam,
            float iAlpha = DEFAULT_ALPHA,
            float iBeta1 = DEFAULT_BETA1,
            float iBeta2 = DEFAULT_BETA2,
            float iTargetCost = DEFAULT_TARGET_COST,
            int iBatchSize = DEFAULT_BATCH_SIZE,
            int iMaxEpoch = DEFAULT_MAX_EPOCH
            )
        {
            Optimizer = iOptimizer;
            Alpha = iAlpha;
            Beta1 = iBeta1;
            Beta2 = iBeta2;
            TargetCost = iTargetCost;
            BatchSize = iBatchSize;
            MaxEpoch = iMaxEpoch;
        }

        // copy ctor
        public TrainingParams (TrainingParams other)
        {
            Optimizer = other.Optimizer;
            Alpha = other.Alpha;
            Beta1 = other.Beta1;
            Beta2 = other.Beta2;
            TargetCost = other.TargetCost;
            BatchSize = other.BatchSize;
            MaxEpoch = other.MaxEpoch;
        }

        public override String ToString ()
        {
            String str = "Optimizer: " + Enum.GetName(typeof(Optimizers), Optimizer);
            str += "\tAlpha: " + Alpha;
            str += "\tBeta1: " + Beta1;
            str += "\tBeta2: " + Beta2;
            str += "\tTarget Cost: " + TargetCost;
            str += "\tBatchSize: " + BatchSize;
            str += "\tMaxEpoch: " + MaxEpoch;

            return str;
        }

    }// TrainingParams{}

    class MNISTOptimiser
    {
        // number of images in each of the respective MNIST datasets
        private const int NUM_TRAINING_IMAGES = 60000;
        private const int NUM_TEST_IMAGES = 10000;

        /// <summary>
        /// Creates an input matrix with normalised values from an MNIST image
        /// and generates the corresponding expected output matrix
        /// </summary>
        /// <param name="img"></param>
        /// <param name="input"> input matrix to adjust </param>
        /// <param name="target"> target matrix to adjust </param>
        public static void ProcessImg(Image img, out Matrix input, out Matrix target)
        {
            float[] vals = new float[28 * 28];

            // normalise input data
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    vals[i * 28 + j] = img.data[i, j] / 255f;
                }
            }

            input = new Matrix(28 * 28, 1, vals);
            target = new Matrix(10, 1);
            target.SetValue(img.label, 0, 1.0f);
        }

        public static float TrainNetwork (NeuralNetwork net, TrainingParams tParams = null, string logFileName = null )
        {
            // if no parameters are specified, use the defaults
            if (tParams == null)
                tParams = new TrainingParams();

            int startTop = 0;

            // number of batches to run before logging
            const int numStepsPerLog = 10;
            int MOVING_AVERAGE_SIZE = 100;

            bool doneTraining = false;
            
            // read MNIST data
            var images = MNISTReader.ReadTrainingData();

            // use default values for training if training parameters weren't specified
            if (tParams == null)
                tParams = new TrainingParams(); 

            int NUM_BATCHES = NUM_TRAINING_IMAGES / tParams.BatchSize;

            float endCost = 0;

            while (!doneTraining)
            {
                int epochCount = 1;

                Matrix[] vdW = new Matrix[net.Layers.Count];
                Matrix[] vdB = new Matrix[net.Layers.Count];
                Matrix[] sdW = new Matrix[net.Layers.Count];
                Matrix[] sdB = new Matrix[net.Layers.Count];

                
                // for logging
                float costSum = 0;
                float accuracySum = 0;
                Queue<float> costQueue = new Queue<float>(MOVING_AVERAGE_SIZE);
                Queue<float> accuracyQueue = new Queue<float>(MOVING_AVERAGE_SIZE);
                
                List<string> logData = new List<string>();

                float alpha = tParams.Alpha;
                const float decay = 0.05f;

                // run through the epoch
                while (epochCount <= tParams.MaxEpoch && !doneTraining)
                {
                    // shuffle training data and get enumerator
                    images = images.Shuffle ();
                    IEnumerator<Image> imgEnumerator = images.GetEnumerator();

                    int batchCount = 1;

                    int idx = 0;
                    foreach (DenseLayer l in net.Layers)
                    {
                        vdW[idx] = new Matrix(l.Weights.Rows, l.Weights.Cols);
                        sdW[idx] = new Matrix(l.Weights.Rows, l.Weights.Cols);
                        vdB[idx] = new Matrix(l.Biases.Rows, l.Biases.Cols);
                        sdB[idx] = new Matrix(l.Biases.Rows, l.Biases.Cols);
                        idx++;
                    }

                    Matrix[] weightChanges = new Matrix[net.Layers.Count];
                    Matrix[] biasChanges = new Matrix[net.Layers.Count];

                    while (batchCount <= NUM_BATCHES)
                    {    
                        float batchAvgCost = 0;
                        float batchAccuracy = 0;

                        // weight and bias derivatives for all network layers
                        Matrix[] batchWeightGradients;
                        Matrix[] batchBiasGradients;

                        // run training batch
                        RunTrainingBatch (
                            ref net, 
                            ref imgEnumerator, 
                            tParams,
                            out batchAccuracy,
                            out batchAvgCost,
                            out batchWeightGradients,
                            out batchBiasGradients);

                        switch (tParams.Optimizer)
                        {
                            // update the weights using the mini-batch gradient
                            case Optimizers.Batch:
                                for (int i = 0; i < net.Layers.Count; i ++ )
                                {
                                    weightChanges[i] = -tParams.Alpha * batchWeightGradients[i];
                                    biasChanges[i] = -tParams.Alpha * batchBiasGradients[i];
                                 }
                                break;

                            case Optimizers.Momentum:
                                {
                                    // calculate momentum for all layers
                                    for (int i = 0; i < batchWeightGradients.Length; i++)
                                    {
                                        vdW[i] = tParams.Beta1 * vdW[i] + (1.0f - tParams.Beta1) * batchWeightGradients[i];
                                        vdB[i] = tParams.Beta1 * vdB[i] + (1.0f - tParams.Beta1) * batchBiasGradients[i];

                                        weightChanges[i] = -tParams.Alpha * vdW[i];
                                        biasChanges[i] = -tParams.Alpha * vdB[i];
                                    }

                                    break;
                                }

                            case Optimizers.RMSProp:

                                for (int i = 0; i < net.Layers.Count; i ++)
                                {
                                    sdW[i] = tParams.Beta2 * sdW[i] + (1.0f - tParams.Beta2) * batchWeightGradients[i].ElementWisePow(2);
                                    sdB[i] = tParams.Beta2 * sdB[i] + (1.0f - tParams.Beta2) * batchBiasGradients[i].ElementWisePow(2);

                                    weightChanges[i] = new Matrix(net.Layers[i].Weights.Rows, net.Layers[i].Weights.Cols);
                                    biasChanges[i] = new Matrix(net.Layers[i].Biases.Rows, net.Layers[i].Biases.Cols);

                                    for (int j = 0; j < sdW[i].values.Length; j ++)
                                    {
                                        weightChanges[i].values[j] = -tParams.Alpha * batchWeightGradients[i].values[j] / ((float)Math.Sqrt (sdW[i].values[j] + 10e-8f) );
                                    }
                                    for (int j = 0; j < sdB[i].values.Length; j ++)
                                    {
                                        biasChanges[i].values[j] = -tParams.Alpha * batchBiasGradients[i].values[j] / ((float)Math.Sqrt(sdB[i].values[j] + 10e-8f) );
                                    }

                                }

                                break;

                            case Optimizers.Adam:

                                float b1Norm = 1.0f; // / (1 - (float)Math.Pow(tParams.Beta1, batchCount));
                                float b2Norm = 1.0f; // / (1 - (float)Math.Pow(tParams.Beta2, batchCount));

                                for (int i = 0; i < net.Layers.Count; i++)
                                {
                                    vdW[i] = tParams.Beta1 * vdW[i] + (1.0f - tParams.Beta1) * batchWeightGradients[i];
                                    vdB[i] = tParams.Beta1 * vdB[i] + (1.0f - tParams.Beta1) * batchBiasGradients[i];

                                    sdW[i] = tParams.Beta2 * sdW[i] + (1.0f - tParams.Beta2) * batchWeightGradients[i].ElementWisePow(2);
                                    sdB[i] = tParams.Beta2 * sdB[i] + (1.0f - tParams.Beta2) * batchBiasGradients[i].ElementWisePow(2);

                                    vdW[i] = b1Norm * vdW[i];
                                    vdB[i] = b1Norm * vdB[i];
                                    sdW[i] = b2Norm * sdW[i];
                                    sdB[i] = b2Norm * sdB[i];

                                    weightChanges[i] = new Matrix(net.Layers[i].Weights.Rows, net.Layers[i].Weights.Cols);
                                    biasChanges[i] = new Matrix(net.Layers[i].Biases.Rows, net.Layers[i].Biases.Cols);

                                    for (int j = 0; j < sdW[i].values.Length; j++)
                                    {
                                        weightChanges[i].values[j] = -alpha * vdW[i].values[j] / ((float)Math.Sqrt(sdW[i].values[j] + 10e-8f));
                                    }
                                    for (int j = 0; j < sdB[i].values.Length; j++)
                                    {
                                        biasChanges[i].values[j] = -alpha * vdB[i].values[j] / ((float)Math.Sqrt(sdB[i].values[j] + 10e-8f));
                                    }

                                }

                                break;
                        }

                        // apply weight and bias changes to network
                        for (int i = 0; i < net.Layers.Count; i ++)
                        {
                            net.Layers[i].Weights += weightChanges[i];
                            net.Layers[i].Biases += biasChanges[i];
                        }

                        // populate cost queue for moving average
                        if (costQueue.Count == MOVING_AVERAGE_SIZE)
                        {
                            costSum -= costQueue.Dequeue();
                            accuracySum -= accuracyQueue.Dequeue();
                        }

                        costSum += batchAvgCost;
                        accuracySum += batchAccuracy;

                        costQueue.Enqueue(batchAvgCost);
                        accuracyQueue.Enqueue(batchAccuracy);
                        
                        float epochAvgAccuracy = accuracySum / accuracyQueue.Count; 
                        float epochAvgCost = costSum / costQueue.Count;
                        endCost = epochAvgCost;

                        // reached training goal
                        if (epochAvgCost <= tParams.TargetCost)
                        {
                            UI.DisplayInfo($"Done training! Target {tParams.TargetCost}\tCost average over epoch{epochAvgCost}");
                            doneTraining = true;
                            break;
                        }

                        if (logFileName != null)
                        {
                            // reset cursor
                            Console.SetCursorPosition(0, startTop);

                            // display training metrics
                            Console.Write($"epoch ({epochCount} / {tParams.MaxEpoch})");
                            UI.DisplayProgressBar((float)batchCount / (float)NUM_BATCHES, 10);

                            //Console.WriteLine($"[ Average Accuracy ] {epochAvgAccuracy}\t[ Average Cost ] {epochAvgCost}".PadRight (Console.WindowWidth-2));
                            //Console.WriteLine($"[ Accuracy ] {batchAccuracy}\t[ Cost ] {batchAvgCost}".PadRight (Console.WindowWidth-2));
                            //Console.WriteLine($"[ Current alpha ] {alpha}".PadRight (Console.WindowWidth-2));
                        }

                        if (logFileName != null) // check if log file is supplied
                        {
                            int step = batchCount + (NUM_BATCHES * (epochCount - 1));
                            logData.Add(step.ToString());
                            logData.Add(batchAccuracy.ToString ());
                            logData.Add(batchAvgCost.ToString ());
                            logData.Add(epochAvgAccuracy.ToString ());
                            logData.Add(epochAvgCost.ToString());

                            if (batchCount % numStepsPerLog == 0)
                            {
                                StreamWriter loggingStreamWriter = File.AppendText(logFileName);

                                LogTrainingData(loggingStreamWriter, logData.ToArray(), 5);
                                
                                logData = new List<string>();
                                loggingStreamWriter.Close();
                            }
                        }    

                        batchCount++; 
                    }//batchloop

                    alpha = alpha * 1.0f / (1.0f + decay);
                    startTop++;
                    epochCount++;
                }//epochLoop

                // check if trained to target cost
                doneTraining = true;
            }//TrainingLoop    

            return endCost;

        }//TrainNetwork ()

        /// <summary>
        /// Runs a training batch within a training epoch. Performs weight and bias optimisation using backpropagation 
        /// and a square difference cost function.
        /// </summary>
        /// <param name="net"></param>
        /// <param name="imgEnumerator"></param>
        /// <param name="batchSize">number of training examples to train over in this batch</param>
        /// <param name="numCorrect">number of examples correctly classified</param>
        /// <param name="avgCost">average output cost of the network over the batch</param>
        private static void RunTrainingBatch (
            ref NeuralNetwork net, 
            ref IEnumerator<Image> imgEnumerator, 
            TrainingParams tParams,
            out float accuracy,
            out float avgCost,
            out Matrix[] weightDeltas,
            out Matrix[] biasDeltas)
        {
            double costSum = 0;

            int numCorrect = 0;
            int count = 1;

            // initialise buffers for weight and bias deltas storage
            Matrix[] accumulatedWeightDeltas    = new Matrix[net.Layers.Count];
            Matrix[] accumulatedBiasDeltas      = new Matrix[net.Layers.Count];

            for ( int i = 0; i < net.Layers.Count; i ++ )
            {
                DenseLayer l = net.Layers[i];
                accumulatedWeightDeltas[i]  = new Matrix(l.Weights.Rows, l.Weights.Cols);
                accumulatedBiasDeltas[i]    = new Matrix(l.Biases.Rows, l.Biases.Cols);
            }

            // iterate through training images while still within the current batch
            while (imgEnumerator.MoveNext () && count <= tParams.BatchSize)
            {   
                // calculate input and output of the neural network based on training example
                Matrix input;
                Matrix expected;
                ProcessImg(imgEnumerator.Current, out input, out expected);

                // feed forward input to get the networks output for training example
                Matrix prediction = net.Predict(input);

                // check result
                if (Matrix.ArgMaxIndex (prediction) == imgEnumerator.Current.label)
                    numCorrect++;

                // calculate cost for this training example to quantify how wrong the network is
                float cost = net.GetCost(expected);
                costSum += cost;

                // derivative of Cost w.r.t
                Matrix dCdA = net.GetCostDerivative(expected);

                // propagate error back through network
                for (int i = net.Layers.Count - 1; i >= 0; i --)
                {
                    DenseLayer l = net.Layers[i];

                    // calculate current layer delta 
                    Matrix del = l.BackProp(dCdA);

                    // calculate weight and bias derivatives
                    Matrix dW = del * ((i==0)? net.Inputs : net.Layers[i-1].A).Transpose();
                    Matrix dB = del;

                    accumulatedWeightDeltas[i] += (1.0f / (float)tParams.BatchSize) * dW;
                    accumulatedBiasDeltas[i] += (1.0f / (float)tParams.BatchSize) * dB;

                    if ( i > 0 )
                        dCdA = l.Weights.Transpose() * del;
                }

                count++;
            }//batch loop

            weightDeltas = accumulatedWeightDeltas;
            biasDeltas = accumulatedBiasDeltas;

            // calculate metrics
            avgCost = (float)costSum / (float)count;
            accuracy = (float)numCorrect / (float)count;

        }//RunTrainingBatch

        public static float TestNetwork (NeuralNetwork net)
        {
            var images = MNISTReader.ReadTestData();

            int counter = 1;
            int numCorrect = 0;

            double costSum = 0;

            Console.Clear();
            Console.SetCursorPosition(0, 0);

            var enumerator = images.GetEnumerator();

            while (enumerator.MoveNext ())
            {
                var img = enumerator.Current;

                Matrix input;
                Matrix expected;
                ProcessImg(img, out input, out expected);

                Matrix result = net.Predict(input);
                costSum += net.GetCost(expected);

                if (Matrix.ArgMaxIndex (result) == img.label)
                    numCorrect++;

                if (counter % (NUM_TEST_IMAGES / 30) == 0)
                {
                    UI.DisplayInfo("Testing...");
                    UI.DisplayProgressBar((float)counter / (float) NUM_TEST_IMAGES);

                    if (counter < NUM_TEST_IMAGES)
                        Console.SetCursorPosition(0, 0);
                }

                counter++;
            }
            
            return (float)numCorrect / (float)counter;
        }


        /// <summary>
        /// Appends to a log file the gradient descent step data provided
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="step"></param>
        /// <param name="cost"></param>
        /// <param name="accuracy"></param>
        private static void LogTrainingData (StreamWriter sw, string [] data, int numDataPerRow)
        {
            int ctr = 1;
            foreach (string d in data)
            {
                if (ctr == 1)
                    sw.Write(d);
                else if (ctr == numDataPerRow)
                {
                    sw.WriteLine("," + d);
                    ctr = 0;
                }
                else
                    sw.Write("," + d);
                
                ctr++;
            }
        }
    }
}