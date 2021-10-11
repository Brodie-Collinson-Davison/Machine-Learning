using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

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
        /// Tests the given neural network over the entire test dataset
        /// </summary>
        /// <param name="net"></param>
        /// <returns> accuracy of the network </returns>
        public static float TestNetwork(NeuralNetwork net)
        {
            // read test data
            var images = MNISTReader.ReadTestData();

            float accuracy = 0.0f;
            int count = 1;
            int correct = 0;

            // magic number
            // specifies the number of images to test between displaying current accuracy
            const int upd_freq = 100;

            // iterate over test data
            foreach (var img in images)
            {
                // prep image
                Matrix input = null;
                Matrix target = null;
                MNISTOptimiser.processImg(img, out input, out target);

                // get prediction
                Matrix prediction = net.predict(input);
                float highestOutput = 0.0f;
                int prediction_int = 0;
                for (int i = 0; i < prediction.Rows; i++)
                {
                    if (prediction.GetValue(i, 0) > highestOutput)
                    {
                        highestOutput = prediction.GetValue(i, 0);
                        prediction_int = i;
                    }
                }

                // check if prediction is correct
                if (prediction_int == img.label)
                    correct++;

                // display testing progress
                if (count % upd_freq == 0)
                {
                    // progress bar
                    UI.DisplayProgressBar((float)count / (float)NUM_TEST_IMAGES);
                    Console.Write("Correct: {0:000} / Total: {1}\t ||\tAccuracy = {2:0.00}%", correct, count, ((float)correct / (float)count) * 100);
                    Console.CursorLeft = 0;
                    Console.CursorTop -= 2;
                }

                count++;
            }

            // calculate accuracy
            accuracy = (float)correct / (float)count;
            accuracy *= 100;

            return accuracy;
        }

        /// <summary>
        /// Train a neural network using momentum based batched gradient descent
        /// </summary>
        /// <param name="net"> network to train </param>
        /// <param name="trainingParams"></param>
        /// <param name="logFileName"> if specified the training will be logged to this file </param>
        public static void TrainNetwork(ref NeuralNetwork net, TrainingParams trainingParams, string logFileName = null)
        {
            // read training data
            var data = MNISTReader.ReadTrainingData();

            bool doneTraining = false;
            int numBatchesPerEpoch = NUM_TRAINING_IMAGES / trainingParams.BatchSize;
            int numCorrect = 0;
            int batchCount = 1;
            int stepCount = 0;
            int epochCount = 1;

            double costSum = 0;

            // train network
            while (!doneTraining) // Training loop
            {
                // display epoch number before running epoch
                Console.Clear();
                UI.SkpLn(2);
                UI.DisplayCentered($"Current Epoch: {epochCount} / {trainingParams.MaxEpoch}");

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

                    // perform a gradient descent step over a batch
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
                    Console.SetCursorPosition(0, 3);

                    // Epoch progress bar
                    UI.DisplayCentered ($"{batchCount} / {numBatchesPerEpoch}");
                    UI.DisplayProgressBar(epochCompletion);

                    UI.SkpLn(5);

                    // Epoch output
                    UI.DisplayCentered("EPOCH");
                    Console.WriteLine("Avg Cost: {0:F5}", costSum / (double)batchCount);
                    Console.WriteLine("Accuracy: {0:F1}%", 100.0f * (float)numCorrect / (float)(batchCount * trainingParams.BatchSize));

                    UI.SkpLn(2);

                    // Batch output
                    UI.DisplayCentered("BATCH");
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
                    doneTraining = true;

                // check if reached epoch limit
                if (epochCount >= trainingParams.MaxEpoch)
                    doneTraining = true;

                // reset epoch counters
                costSum = 0;
                batchCount = 1;
                numCorrect = 0;
                epochCount++;
            
            }//training loop
        }//TrainNetwork ()

        /// <summary>
        /// Run a batch of predictions to perform a gradient descent step
        /// </summary>
        /// <param name="net"></param>
        /// <param name="trainingParams"></param>
        /// <param name="imgEnumerator"> reference to the enumerable list of Images </param>
        /// <param name="avgCost"></param>
        /// <param name="accuracy"></param>
        /// <param name="numCorrect"></param>
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

            // iterate through training data until EOF or batch size reached
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

                // sum output cost to get total network cost for the example
                float sum = 0;
                for (int i = 0; i < net.Layers[net.Layers.Length - 1].Size; i++)
                {
                    sum += cost.GetValue(i, 0);
                }
                costSum += sum;

                // accumulate weight and bias changes
                for (int i = net.getNumLayers() - 1; i >= 0; i--)
                {
                    // delW(l) = layerDelta(l) * activations(l-1)[T]
                    // delB(l) = layerDelta(l)
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
        /// Represents the predicted of a classifier
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

        /// <summary>
        /// Appends to a log file the gradient descent step data provided
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="step"></param>
        /// <param name="cost"></param>
        /// <param name="accuracy"></param>
        static void LogTrainingData(string fileName, int step, double cost, double accuracy)
        {
            String data = string.Format($"{step},{cost},{accuracy}");
            FileManager.AppendToFile(fileName, data);
        }
    }
}