using System;
using System.IO;
using System.Collections.Generic;

//Note data ranges from 0 to 255 as a brightness value per pixel
namespace NN
{
    public class MNISTReader
    {
        private const string TRAIN_lABELS_FILE_NAME = "MNIST_Dataset/train-labels.idx1-ubyte";
        private const string TRAIN_IMAGES_FILE_NAME = "MNIST_Dataset/train-images.idx3-ubyte";
        private const string TEST_IMAGES_FILE_NAME = "MNIST_Dataset/t10k-images.idx3-ubyte";
        private const string TEST_LABELS_FILE_NAME = "MNIST_Dataset/t10k-labels.idx1-ubyte";

        /// <summary>
        /// Ensures the MNIST dataset is accessible
        /// </summary>
        /// <returns> true if can access </returns>
        public static bool CanAccessMNISTData ()
        {
            bool success = false;

            try
            {
                // check if directory exists
                if (Directory.Exists(Path.Combine(Directory.GetCurrentDirectory(), "MNIST_Dataset")))
                    success = true;
            }
            catch (Exception e)
            {
                UI.Display_Error(e.Message);
            }

            return success;
        }

        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(TRAIN_IMAGES_FILE_NAME, TRAIN_lABELS_FILE_NAME))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(TEST_IMAGES_FILE_NAME, TEST_LABELS_FILE_NAME))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            // open file streams
            FileStream labelFStream = new FileStream(labelsPath, FileMode.Open);
            FileStream imagesFStream = new FileStream(imagesPath, FileMode.Open);

            // create binary readers
            BinaryReader labels = new BinaryReader(labelFStream);
            BinaryReader images = new BinaryReader(imagesFStream);

            // read data as per instructions on: http://yann.lecun.com/exdb/mnist/
            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new byte[height, width];

                arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                yield return new Image()
                {
                    data = arr,
                    label = labels.ReadByte()
                };
            }

            labelFStream.Close();
            imagesFStream.Close();
        }
    }

    // class to contain MNIST images
    public class Image
    {
        public byte label { get; set; }
        public byte[,] data { get; set; }
    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
}