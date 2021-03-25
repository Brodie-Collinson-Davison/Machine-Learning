using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace NN
{
    class TrainingParams
    {
        // default training parameter values
        private const float DEFAULT_LEARNING_RATE = 0.01f;
        private const float DEFAULT_MOMENTUM = 0.1f;
        private const float DEFAULT_TARGET_COST = 0.05f;
        private const int DEFAULT_BATCH_SIZE = 150;
        private const int DEFAULT_MAX_EPOCH = 10;

        // class properties

        public float LearningRate { get; set; }
        public float Momentum { get; set; }
        public float TargetCost { get; set; }
        public int BatchSize { get; set; }
        public int MaxEpoch { get; set; }

        // constructors

        // default constructor
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

    }// TrainingParams{}

    class MNISTOptimiser
    {
        // number of test images in the MNIST dataset
        private const int NUM_TRAINING_IMAGES = 60000;
        private const int NUM_TEST_IMAGES = 10000;

        public static void TrainNetwork(ref NeuralNetwork net, TrainingParams trainingParams, string logFileName = null)
        {
            // read training data
            var data = MNISTReader.ReadTrainingData();

            bool doneTraining = false;
            int numBatchesPerEpoch = NUM_TRAINING_IMAGES / trainingParams.BatchSize;
            int numCorrect = 0;
            int batchCount = 1;
            int stepCount = 0;
            int epochCount = 0;

            double costSum = 0;

            Stopwatch watch = new Stopwatch();

            // train network
            while (!doneTraining) // Training loop
            {
                // display epoch number
                Console.SetCursorPosition(0, 0);
                Console.WriteLine("Epoch: " + epochCount);

                // shuffle data at each epoch
                data = data.Shuffle();

                // get new enumerator through shuffled data
                IEnumerator<Image> imgEnumerator = data.GetEnumerator();

                // Epoch loop
                while (batchCount <= (NUM_TRAINING_IMAGES / trainingParams.BatchSize))
                {
                    int numCorrectThisBatch;
                    double avgCost;
                    double accuracy;
                    float epochCompletion = (float)batchCount / (float)numBatchesPerEpoch;

                    // perform a gradient descent step over a mini batch
                    RunBatch
                        (
                        ref net,
                        trainingParams,
                        ref imgEnumerator,
                        out avgCost,
                        out accuracy,
                        out numCorrectThisBatch
                        );

                    numCorrect += numCorrectThisBatch;

                    // reset cursor position to write stats
                    Console.SetCursorPosition(0, 1);

                    // Epoch progress bar
                    int barSize = 25;
                    Console.WriteLine("{0} / {1}", batchCount, numBatchesPerEpoch);
                    Console.Write("[");
                    for (int i = 0; i < barSize; i++)
                    {
                        if ((float)i / (float)barSize < epochCompletion)
                            Console.Write("#");
                        else
                            Console.Write(" ");
                    }
                    Console.WriteLine("]\n");

                    // Epoch output
                    Console.WriteLine("\tEpoch:");
                    Console.WriteLine("Avg Cost: {0:F5}", costSum / (double)batchCount);
                    Console.WriteLine("Accuracy: {0:F1}%", 100.0f * (float)numCorrect / (float)(batchCount * trainingParams.BatchSize));

                    Console.WriteLine("");

                    // Batch output
                    Console.WriteLine("\tBatch:");
                    Console.WriteLine("Avg Cost: {0:F3}\nAccuracy: {1:F1}", avgCost, 100.0 * accuracy);

                    // logging
                    if (logFileName != null)
                    {
                        LogTrainingData(logFileName, stepCount, avgCost, accuracy);
                    }

                    batchCount++;
                    stepCount++;
                    costSum += avgCost;
                }

                // calculate average cost over entire epoch
                double epochAvgCost = costSum / (double)numBatchesPerEpoch;

                // check if target cost is reached
                if (epochAvgCost <= trainingParams.TargetCost)
                    //  stop training
                    doneTraining = true;

                // check if reached epoch limit
                if (epochCount >= trainingParams.MaxEpoch)
                    doneTraining = true;

                // reset epoch counters
                costSum = 0;
                batchCount = 1;
                numCorrect = 0;
                epochCount++;
            }
        }

        public static void RunBatch
            (
            ref NeuralNetwork net,
            TrainingParams trainingParams,
            ref IEnumerator<Image> imgEnumerator,
            out double avgCost,
            out double accuracy,
            out int numCorrect
            )
        {
            numCorrect = 0;
            int count = 1;
            double costSum = 0;

            // initialise bias and weight delta storage
            Matrix[] accumulatedWeightChanges = new Matrix[net.getNumLayers()];
            Matrix[] accumulatedBiasChanges = new Matrix[net.getNumLayers()];
            Matrix[] prevWeightChanges = new Matrix[net.getNumLayers()];
            Matrix[] prevBiasChanges = new Matrix[net.getNumLayers()];
            for (int i = 0; i < net.getNumLayers(); i++)
            {
                Layer l = net.Layers[i];
                accumulatedWeightChanges[i] = new Matrix(l.WM.Rows, l.WM.Cols);
                prevWeightChanges[i] = new Matrix(l.WM.Rows, l.WM.Cols);
                accumulatedBiasChanges[i] = new Matrix(l.BM.Rows, 1);
                prevBiasChanges[i] = new Matrix(l.BM.Rows, 1);
            }

            // iterate through training data untill data end reached or batch size reached
            while (imgEnumerator.MoveNext() && count < trainingParams.BatchSize)
            {
                // prepare training example
                Matrix input = null;
                Matrix output = null;
                processImg(imgEnumerator.Current, out input, out output);

                // calculate cost and deltas
                Matrix cost = null;
                Matrix[] exampleOutput = null;
                Matrix[] exampleDeltas = net.backProp(input, output, out exampleOutput, out cost);

                // cost
                float sum = 0;
                for (int i = 0; i < 10; i++)
                {
                    sum += cost.GetValue(i, 0);
                }
                costSum += sum;

                // accumulate weighta and bias changes
                // delW(l) = layerDelta(l) * activations(l-1)
                // delB(l) = layerDelta(l)
                for (int i = net.getNumLayers() - 1; i >= 0; i--)
                {
                    Matrix dW = exampleDeltas[i] * exampleOutput[i].Transpose();
                    Matrix dB = exampleDeltas[i];

                    // add momentum term using last changes
                    dW += trainingParams.Momentum * prevWeightChanges[i];
                    dB += trainingParams.Momentum * prevBiasChanges[i];

                    // set past changes for momentum gradient descent
                    prevWeightChanges[i] = dW;
                    prevBiasChanges[i] = dB;

                    // stage weight and bias changes to be added
                    accumulatedWeightChanges[i] += dW;
                    accumulatedBiasChanges[i] += dB;
                }

                // check if prediction is correct
                int prediction_int = GetHighestOutputIndex(net.predict(input));
                if (prediction_int == imgEnumerator.Current.label)
                {
                    numCorrect++;
                }
                count++;
            }

            // apply accumulated weight and bias changes
            // use accumulated changes and learning rate to get updated weights and biases
            for (int i = 0; i < net.getNumLayers(); i++)
            {
                Matrix updW = net.Layers[i].WM - trainingParams.LearningRate * accumulatedWeightChanges[i];
                Matrix updB = net.Layers[i].BM - trainingParams.LearningRate * accumulatedBiasChanges[i];

                net.Layers[i].WM = updW;
                net.Layers[i].BM = updB;
            }

            // set out variables
            avgCost = costSum / (double)trainingParams.BatchSize;
            accuracy = (double)numCorrect / (double)trainingParams.BatchSize;
        }

        /// <summary>
        /// Creates an input matrix with normalised values from an MNIST image
        /// and generates the corresponding expected output matrix
        /// </summary>
        /// <param name="img"></param>
        /// <param name="input"> input matrix to adjust </param>
        /// <param name="target"> target matrix to adjust </param>
        public static void processImg(Image img, out Matrix input, out Matrix target)
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

        /// <summary>
        /// Returns the index of the highest output value.
        /// Represents the predicted digit 
        /// </summary>
        /// <param name="prediction"> Network output matrix </param>
        /// <returns></returns>
        public static int GetHighestOutputIndex(Matrix prediction)
        {
            int prediction_int = 0;

            float highestOutput = 0.0f;
            for (int i = 0; i < prediction.Rows; i++)
            {
                if (prediction.GetValue(i, 0) > highestOutput)
                {
                    highestOutput = prediction.GetValue(i, 0);
                    prediction_int = i;
                }
            }

            return prediction_int;
        }

        static void LogTrainingData(string fileName, int step, double cost, double accuracy)
        {
            string data = string.Format($"{step},{cost},{accuracy}");
            FileManager.AppendToFile(fileName, data);
        }
    }
}