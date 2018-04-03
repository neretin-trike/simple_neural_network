﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.Statistics;
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

        public static Matrix<double> binaryCrossEnetropy(Matrix<double> X, Matrix<double> Y, Matrix<double> sync)
        {
            Matrix<double> J = Matrix<double>.Build.Random(15, 4);

            var m = X.ColumnCount * X.RowCount;

            foreach (Tuple<int, int, double> x in X.Storage.EnumerateIndexed())
            {
                
                var a = Y[x.Item1, x.Item2] * Matrix.Log(nonlin(X*sync));
                var b = (1 - Y[x.Item1, x.Item2]) * Matrix.Log(1 - nonlin(X * sync));
                J += a + b;

                //var a = y.Item3 * Matrix.Log(nonlin(X[y.Item1, y.Item2] * sync));
                //var b = Matrix.Log(1 - nonlin(X[y.Item1, y.Item2] * sync));
                //J += -(1 / m) * (a + (1 - y.Item3) * b);
            }

            J = -(1 / m) * J;

            return J;
        }

        static void Main(string[] args)
        {
            
            Matrix<double> x = DenseMatrix.OfArray(new double[,] {
                                    {0.095,0.015,0.02},
                                    {0.095,0.070,0.002},
                                    {0.065,0.010,0.003},
                                    {0.012,0.025,0.005},
                                    {0.03, 0.012, 0.06},
                                    {0.065, 0.05, 0.003},
                                    {0.045, 0.07, 0.02},
                                    {0.1, 0.02, 0.065},
                                    {0.007, 0.008, 0.010},
                                    {0.003, 0.004, 0.005},
                                    {0.015, 0.012, 0.006},
                                    {0.32, 0.089, 0.02},
                                    {0.05, 0.02, 0.1},
                                    {0.12, 0.04, 0.02},
                                    {0.22, 0.06, 0.02} });

            Matrix<double> y = DenseMatrix.OfArray(new double[,] {
                                    {1, 0, 0, 0, 0},
                                    {1, 0, 0, 0, 0},
                                    {0, 1, 0, 0, 0},
                                    {0, 0, 1, 0, 0},
                                    {0, 0, 0, 0, 1},
                                    {0, 0, 1, 0, 0},
                                    {0, 0, 1, 0, 0},
                                    {0, 0, 0, 0, 1},
                                    {0, 1, 0, 0, 0},
                                    {0, 1, 0, 0, 0},
                                    {0, 1, 0, 0, 0},
                                    {0, 1, 0, 0, 0},
                                    {0, 0, 0, 1, 0},
                                    {1, 0, 0, 0, 0},
                                    {0, 0, 0, 1, 0} });

            //Matrix<double> y = DenseVector.OfArray(new double[] { 0, 1, 1, 0 }).ToColumnMatrix();

            Matrix<double> syn0 = 2 * Matrix<double>.Build.Random(3, 6, 1) - 1;
            Matrix<double> syn1 = 2 * Matrix<double>.Build.Random(6, 5, 1) - 1;

            Matrix<double> l1 = Matrix<double>.Build.Random(1,1);
            Matrix<double> l2 = Matrix<double>.Build.Random(1,1);

            for (var i = 0; i < 100000; i++)
            {
                var l0 = x;

                l1 = nonlin(l0 * syn0);
                l2 = nonlin(l1 * syn1);

                var l2_error = y - l2;

                if (i % 10000 == 0) 
                    Console.WriteLine( "Error: {0}", Statistics.Mean(l2_error.Storage.Enumerate()));

                var refer2 = nonlin(l1 * syn1);
                // Метод наименьших квадратов (квадратичная ошибка)
                //var l2_delta = l2_error.PointwiseMultiply( binaryCrossEnetropy(l1,y,syn1));
                var l2_delta = l2_error.PointwiseMultiply(nonlin(refer2, true));

                var l1_error = l2_delta * syn1.Transpose();
                var refer1 = nonlin(l0 * syn0);
                // Метод наименьших квадратов (квадратичная ошибка)
                //var l1_delta = l1_error.PointwiseMultiply(binaryCrossEnetropy(l0, y, syn0));
                var l1_delta = l1_error.PointwiseMultiply(nonlin(refer1, true));

                syn1 += l1.Transpose() * l2_delta;
                syn0 += l0.Transpose() * l1_delta;

            }

            //Console.WriteLine("Выходные данные после тренировки:");
            //Console.WriteLine(l1);

            Console.WriteLine("Выходные данные после тренировки:");
            Console.WriteLine(l2);


            Matrix<double> x_test = DenseVector.OfArray(new double[] { 0.31053, 0.04722, 0.13721}).ToColumnMatrix();
            l1 = nonlin(x_test.Transpose() * syn0);
            l2 = nonlin(l1 * syn1);

            Console.WriteLine("Ответ по тестовой выборке:");
            Console.WriteLine(l2);

            Console.ReadKey();
        }
    }
}
