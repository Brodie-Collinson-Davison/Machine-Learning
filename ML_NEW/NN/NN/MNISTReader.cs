using System;
using System.IO;
using System.Collections.Generic;

//Note data ranges from 0 to 255 as a brightness value per pixel
namespace NN
{
    public class MNISTReader
    {
        private const string TrainImages = "dataset/train-images.idx3-ubyte";
        private const string TrainLabels = "dataset/train-labels.idx1-ubyte";
        private const string TestImages = "dataset/t10k-images.idx3-ubyte";
        private const string TestLabels = "dataset/t10k-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(TestImages, TestLabels))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            FileStream labelFStream = new FileStream((Environment.CurrentDirectory + "/../../../" + labelsPath), FileMode.Open);
            FileStream imagesFStream = new FileStream((Environment.CurrentDirectory + "/../../../" + imagesPath), FileMode.Open);

            BinaryReader labels = new BinaryReader(labelFStream);
            BinaryReader images = new BinaryReader(imagesFStream);

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