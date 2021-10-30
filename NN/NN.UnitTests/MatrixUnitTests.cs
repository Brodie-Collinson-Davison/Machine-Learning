using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace NN.UnitTests
{
    [TestClass]
    public class MatrixUnitTests
    {
        [ClassInitialize]
        public static void Setup (TestContext context)
        { 
        }

        [ClassCleanup]
        public static void TearDown ()
        {

        }

        [DataTestMethod]
        [DataRow("1,2;3,4;5,6", 3,2)]
        [DataRow("1,2,3;4,5,6;7,8,9", 3, 3)]
        [DataRow("1,2,3", 1, 3)]
        [TestMethod]
        public void StringConstructor_CorrectArgumentStringGiven_HasCorrectDimensions (string args, int expected_rows, int expected_cols)
        {
            Matrix mat = new Matrix(args);

            Assert.IsTrue(mat.Rows == expected_rows && mat.Cols == expected_cols);
        }

        [DataTestMethod]
        [DataRow ("1;2,2,2,;,3,3")]
        [DataRow("abc,23d,3f;,23")]
        [DataRow("1;;1;1;111;1,2,2")]
        [TestMethod]
        public void StringConstructor_IncorrectArgumentStringGiven_ThrowsException(string args)
        {
            Assert.ThrowsException<ArgumentException>(() => new Matrix(args));
        }

        [TestMethod]
        public void Sqrt_PerformsCorrectCalculation ()
        {
            float[] values = { 0, 1, 2, 3.5f, 25, 36};
            Matrix m = new Matrix(2, 3, values);

            m = m.ElementWiseSqrt();

            for (int i = 0; i < 6; i ++ )
            {
                float expected = (float)Math.Sqrt(values[i]);
                float actual = m.values[i];
                Assert.AreEqual (expected, actual);
            }
        }

        [DataTestMethod]
        [DataRow("1,1,1;2,2,2;3,3,3", "1,2,3;4,5,6;7,8,9", "12,15,18;24,30,36;36,45,54")]
        [DataRow("1,1;2,2;3,3", "1,2,3;1,2,3", "2,4,6;4,8,12;6,12,18")]
        [TestMethod]
        public void OperatorMultiply_MultiplyTwoMatrices_ValuesAreCorrect (string lhs_args, string rhs_args, string expected_args)
        {
            Matrix lhs = new Matrix(lhs_args);
            Matrix rhs = new Matrix(rhs_args);
            Matrix expected = new Matrix(expected_args);

            Matrix actual = lhs * rhs;

            Assert.IsTrue(actual == expected);
        }

        [DataTestMethod]
        [DataRow(3,1, 1,3, 3,3)]
        [DataRow(2, 2, 2, 6, 2, 6)]
        [DataRow(1, 7, 7, 5, 1, 5)]
        [DataRow(1, 1, 1, 1, 1, 1)]
        [TestMethod]
        public void OperatorMultiply_MultiplyTwoMatricesWithCorrectDimensions_ResultHasCorrectDimensions (int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols, int expected_rows, int expected_cols)
        {
            Matrix lhs = new Matrix(lhs_rows, lhs_cols);
            Matrix rhs = new Matrix(rhs_rows, rhs_cols);
            
            Matrix result = lhs * rhs;

            Assert.IsTrue(result.Rows == expected_rows && result.Cols == expected_cols);
        }

        [DataTestMethod]
        [DataRow(3,3,2,2)]
        [TestMethod]
        public void OperatorMultiply_MultiplyTwoMatricesWithIncorrectDimensions_ThrowsException (int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols)
        {
            Matrix lhs = new Matrix(lhs_rows, lhs_cols);
            Matrix rhs = new Matrix(rhs_rows, rhs_cols);

            Assert.ThrowsException<ArgumentException>(() => lhs * rhs);
        }

        // proof of matrix multiplication identity
        [TestMethod]
        public void Identity_MultiplyMatrixByIdentity_ValuesUnchanged ()
        {
            Matrix I = Matrix.Identity(3);
            Matrix M = new Matrix("1,2,3;4,5,6;7,8,9");

            Matrix result = I * M;

            Assert.AreEqual(M, result);
        }
    }
}
