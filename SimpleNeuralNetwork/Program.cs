using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SimpleNeuralNetwork
{
    class Program
    {
        public static Matrix<double> nonlin(Matrix<double> x, bool deriv = false)
        {
            if (deriv == true)
            {
                foreach (Tuple<int, int, double> x2 in x.Storage.EnumerateIndexed())
                {
                    x.Storage[x2.Item1, x2.Item2] = x2.Item3 * (1 - x2.Item3);
                }
                return x;
            }

            foreach (Tuple<int, int, double> x2 in x.Storage.EnumerateIndexed())
            {
                x.Storage[x2.Item1, x2.Item2] = 1 / (1 + Math.Exp(-x2.Item3));
            }
            return x;
        }

        static void Main(string[] args)
        {
            Matrix<double> x = DenseMatrix.OfArray(new double[,] {
                                    {0,0,1},
                                    {0,1,1},
                                    {1,0,1},
                                    {1,1,1} });

            Matrix<double> y = DenseVector.OfArray(new double[] { 0, 0, 1, 1 }).ToColumnMatrix();

            //Matrix<double> syn0 = DenseVector.OfArray(new double[] { -0.59809633, -0.87858146, -0.70332104 }).ToColumnMatrix();
            Matrix<double> syn0 = 2 * Matrix<double>.Build.Random(3, 1, 1) - 1;

            Matrix<double> l1 = Matrix<double>.Build.Random(1,1);

            for (var i = 0; i < 10000; i++)
            {
                var l0 = x;

                l1 = nonlin(l0 * syn0);

                var l1_error = y - l1;

                var refer = nonlin(l0 * syn0);
                var l1_delta = l1_error.PointwiseMultiply(nonlin(refer, true));

                syn0 += l0.Transpose() * l1_delta;

            }

            Console.WriteLine("Выходные данные после тренировки:");
            Console.WriteLine(l1);


            Matrix<double> x_test = DenseVector.OfArray(new double[] { 0.75, 0.85, 0.95 }).ToColumnMatrix();
            l1 = nonlin(x_test.Transpose() * syn0);

            Console.WriteLine("Ответ по тестовой выборке:");
            Console.WriteLine(l1);

            Console.ReadKey();
        }
    }
}
