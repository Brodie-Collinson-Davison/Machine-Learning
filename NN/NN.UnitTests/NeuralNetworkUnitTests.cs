using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace NN.UnitTests
{
    [TestClass]
    public class NeuralNetworkUnitTests
    { 

        [TestMethod]
        public void ArgumentConstructor_CorrectArguments_CreatesCorrectStructure ()
        {
            NeuralNetwork net = new NeuralNetwork("10 5 5 2");

            Assert.AreEqual(10, net.NUM_INPUTS);
            Assert.AreEqual(net.Layers[0].Size, 5);
            Assert.AreEqual(net.Layers[1].Size, 5);
            Assert.AreEqual(net.Layers[2].Size, 2);
        }

        [DataTestMethod]
        [DataRow("13a 10 5s 5s")]
        [DataRow("13s10 5s 5s")]
        [DataRow("13 10 5sr 5s")]
        [DataRow("13a 10 5s5s")]
        [TestMethod]
        public void ArgumentCTOR_IncorrectArguments_ThrowsException ( string args )
        {
            Assert.ThrowsException<ArgumentException>(() => new NeuralNetwork(args));
        }

        [TestMethod]
        public void FeedForward_CorrectInput_CalculatesOutput ()
        {
            float[] vals = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
            Matrix inputs = new Matrix(10, 1, vals);

            NeuralNetwork net = new NeuralNetwork("10 5 5 3");

            Matrix result = net.Predict(inputs);
            Console.WriteLine(result);
            Assert.AreEqual(result.Rows, 3);
        }
    }
}
