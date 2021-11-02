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
            NeuralNetwork net = new NeuralNetwork("10 5 5 2", default);

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
        [DataRow("13 10r 5Ss")]
        [TestMethod]
        public void ArgumentCTOR_IncorrectArguments_ThrowsException ( string args )
        {
            Assert.ThrowsException<ArgumentException>(() => new NeuralNetwork(args, default));
        }

        [TestMethod]
        public void FeedForward_CorrectInput_CalculatesOutput ()
        {
            NeuralNetwork net = new NeuralNetwork("3 5r 5r 2S", default);

            Matrix activations = Matrix.Scrambled(3, 1);
            Matrix l1_w = Matrix.Scrambled (5, 3);
            Matrix l1_b = Matrix.Scrambled (5, 1);
            Matrix l2_w = Matrix.Scrambled (5, 5);
            Matrix l2_b = Matrix.Scrambled (5, 1);
            Matrix l3_w = Matrix.Scrambled (5, 2);
            Matrix l3_b = Matrix.Scrambled (2, 1);

            net.Layers[0].Weights = l1_w;
            net.Layers[0].Biases = l1_b;
            net.Layers[1].Weights = l2_w;
            net.Layers[1].Biases = l2_b;
            net.Layers[2].Weights = l3_w;
            net.Layers[2].Biases = l3_b;

            Matrix tmp = l1_w * activations + l1_b;
        }

        [TestMethod]
        public void SerializationDoesntCompromiseNetwork ()
        {
            NeuralNetwork net = new NeuralNetwork("2 3r 2S", CostFunctions.CCE);

            string json = net.GetJSONString();
            NeuralNetwork deserializedNet = NeuralNetwork.DeserializeFromJSON(json);

            Assert.AreEqual(deserializedNet, net);
        }
    }
}
