using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace NN
{
    public class TrainingParams
    {
        // default training parameter values
        private const float DEFAULT_LEARNING_RATE = 0.01f;
        private const float DEFAULT_MOMENTUM = 0.5f;
        private const float DEFAULT_TARGET_COST = 0.1f;
        private const int DEFAULT_BATCH_SIZE = 150;
        private const int DEFAULT_MAX_EPOCH = 3;

        // class properties

        public float LearningRate { get; set; }
        public float Momentum { get; set; }
        public float TargetCost { get; set; }
        public int BatchSize { get; set; }
        public int MaxEpoch { get; set; }

        // constructors

        // default constructor initialises with default values
        public TrainingParams
            (
            float iLearningRate = DEFAULT_LEARNING_RATE,
            float iMomentum = DEFAULT_MOMENTUM,
            float iTargetCost = DEFAULT_TARGET_COST,
            int iBatchSize = DEFAULT_BATCH_SIZE,
            int iMaxEpoch = DEFAULT_MAX_EPOCH
            )
        {
            LearningRate = iLearningRate;
            Momentum = iMomentum;
            TargetCost = iTargetCost;
            BatchSize = iBatchSize;
            MaxEpoch = iMaxEpoch;
        }

        // copy ctor
        public TrainingParams (TrainingParams other)
        {
            LearningRate = other.LearningRate;
            Momentum = other.Momentum;
            TargetCost = other.TargetCost;
            BatchSize = other.BatchSize;
            MaxEpoch = other.MaxEpoch;
        }

        public override String ToString ()
        {
            String str = "Learning Rate: " + LearningRate;
            str += "\tMomentum: " + Momentum;
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

        public static void TrainNetwork (NeuralNetwork net, TrainingParams tParams = null )
        {
            bool doneTraining = false;
            
            // read MNIST data
            var images = MNISTReader.ReadTrainingData();

            // use default values for training if training parameters weren't specified
            if (tParams == null)
                tParams = new TrainingParams(); 

            int NUM_BATCHES = NUM_TRAINING_IMAGES / tParams.BatchSize;

            Console.WriteLine("Starting Training...");
            Console.WriteLine(tParams);

            // clear log file
            new FileStream("devLogFile.csv", FileMode.Create).Close ();

            while (!doneTraining)
            {
                int epochCount = 1;
                const int numStepsPerLog = 5;

                // run epochs
                while (epochCount <= tParams.MaxEpoch && !doneTraining)
                {
                    // shuffle training data and get enumerator
                    images = images.Shuffle ();
                    IEnumerator<Image> imgEnumerator = images.GetEnumerator();

                    int batchCount = 1;

                    const int numTrackedQuantities = 3;
                    List<string> logData = new List<string>();
                    
                    while (batchCount <= NUM_BATCHES)
                    {
                        float batchAvgCost = 0;
                        float batchAccuracy = 0;

                        // run training batch
                        RunTrainingBatch (
                            ref net, 
                            ref imgEnumerator, 
                            tParams,
                            out batchAccuracy,
                            out batchAvgCost);

                        
                        // reset cursor
                         Console.SetCursorPosition(0, 0);

                        // display training metrics
                        UI.DisplayCentered("Training in Progress");
                        UI.DisplayProgressBar((float)batchCount / (float)NUM_BATCHES);

                        UI.SkpLn(3);

                        Console.WriteLine($"Batch\nAccuracy: {batchAccuracy}\nAvgCost: {batchAvgCost}");

                        int step = batchCount + (NUM_BATCHES * (epochCount - 1));
                        logData.Add(step.ToString());
                        logData.Add(batchAccuracy.ToString ());
                        logData.Add(batchAvgCost.ToString ());

                        if (batchCount % numStepsPerLog == 0)
                        {
                            StreamWriter loggingStreamWriter = File.AppendText("devLogFile.csv");

                            LogTrainingData(loggingStreamWriter, logData.ToArray(), numTrackedQuantities);
                            logData = new List<string>();

                            loggingStreamWriter.Close();
                        }


                        batchCount++; 
                    }

                    epochCount++;
                }//epochLoop

                // check if trained to target cost
                doneTraining = true;
            }//TrainingLoop    

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
            out float avgCost)
        {
            double costSum = 0;

            int numCorrect = 0;
            int count = 1;

            // initialise buffers for weight and bias deltas storage
            Matrix[] accumulatedWeightDeltas    = new Matrix[net.Layers.Count];
            Matrix[] accumulatedBiasDeltas      = new Matrix[net.Layers.Count];
            Matrix[] prevWeightDeltas           = new Matrix[net.Layers.Count];
            Matrix[] prevBiasDeltas             = new Matrix[net.Layers.Count];

            for ( int i = 0; i < net.Layers.Count; i ++ )
            {
                DenseLayer l = net.Layers[i];
                accumulatedWeightDeltas[i]  = new Matrix(l.Weights.Rows, l.Weights.Cols);
                accumulatedBiasDeltas[i]    = new Matrix(l.Biases.Rows, l.Biases.Cols);
                prevWeightDeltas[i]         = new Matrix(l.Weights.Rows, l.Weights.Cols);
                prevBiasDeltas[i]           = new Matrix (l.Biases.Rows, l.Biases.Cols);
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
                float CE = CategoricalCrossEntropyCost (expected, prediction);
                costSum += CE;

                // derivative of Cost w.r.t current layer activations = (y-a)
                Matrix dCdA = (prediction - expected);

                // propagate error back through network
                for (int i = net.Layers.Count - 1; i >= 0; i --)
                {
                    DenseLayer l = net.Layers[i];

                    // calculate current layer delta 
                    Matrix del = l.BackProp(dCdA);

                    // calculate weight and bias derivatives
                    Matrix dW = del * ((i==0)? net.Inputs : net.Layers[i-1].A).Transpose();
                    Matrix dB = del;

                    // add momentum term to deltas
                    if ( count > 1 )
                    {
                        dW += tParams.Momentum * prevWeightDeltas[i];
                        dB += tParams.Momentum * prevBiasDeltas[i];
                    }

                    // set previous changes for momentum algorithm
                    prevWeightDeltas[i] = dW;
                    prevBiasDeltas[i] = dB;

                    // accumulate deltas to gradient approximate
                    accumulatedWeightDeltas[i] += dW;
                    accumulatedBiasDeltas[i] += dB;

                    if ( i > 0 )
                        dCdA = l.Weights.Transpose() * del;
                }

                count++;
            }//batch loop

            // apply stored weight and bias changes
            for (int i = 0; i < net.Layers.Count; i ++ )
            {
                net.Layers[i].Weights = net.Layers[i].Weights - tParams.LearningRate * accumulatedWeightDeltas[i];
                net.Layers[i].Biases = net.Layers[i].Biases - tParams.LearningRate * accumulatedBiasDeltas[i];
            }

            // calculate metrics
            avgCost = (float)costSum / (float)count;
            accuracy = (float)numCorrect / (float)count;

        }//RunTrainingBatch

        public static float TestNetwork (NeuralNetwork net)
        {
            var images = MNISTReader.ReadTestData();

            int counter = 1;
            int numCorrect = 0;

            Console.Clear();

            foreach (var img in images)
            {
                Matrix input;
                Matrix expected;
                ProcessImg(img, out input, out expected);

                Matrix result = net.Predict(input);

                if (Matrix.ArgMaxIndex (result) == img.label)
                    numCorrect++;

                if (counter % (NUM_TEST_IMAGES / 10) == 0)
                {
                    Console.WriteLine("Testing: ");
                    UI.DisplayProgressBar((float)counter / (float) NUM_TEST_IMAGES, 10);

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

        private static float SqrDistOutputCost(Matrix expected, Matrix output)
        {
            Matrix costmat = 0.5f * (output - expected).ElementWisePow(2);
            float costSum = 0;

            // sum all output cost elements to get the total cost of the network
            foreach (float f in costmat.values)
            {
                costSum += f;
            }

            return costSum;
        }

        private static float CategoricalCrossEntropyCost (Matrix expected, Matrix output)
        {
            float f = 0;

            for (int i = 0; i < output.values.Length; i ++)
            {
                f += -expected.values[i] * (float)Math.Log(output.values[i]);
            }

            return f / output.values.Length;
        }
    }
}