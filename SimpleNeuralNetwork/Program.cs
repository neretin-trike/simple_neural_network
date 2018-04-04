using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using Accord.Controls;
using Accord.IO;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord.Statistics.Analysis;

namespace SimpleNeuralNetwork
{
    class Program
    {

        public static Matrix<double> Normalize(Matrix<double> x)
        {
            x = x.NormalizeColumns(1);
            return x;
        }

        public static Matrix<double> Activation(Matrix<double> x)
        {
            foreach (Tuple<int, int, double> x2 in x.Storage.EnumerateIndexed())
            {
                x.Storage[x2.Item1, x2.Item2] = 1 / (1 + Math.Exp(-x2.Item3));
            }
            return x;
        }

        public static Matrix<double> Neuron(Matrix<double> x, Matrix<double> w)
        {
            var z = x * w;
            var output = Activation(z);
            return output;
        }

        //public static Matrix<double> nonlin(Matrix<double> x, bool deriv = false)
        //{
        //    if (deriv == true)
        //    {
        //        foreach (Tuple<int, int, double> x2 in x.Storage.EnumerateIndexed())
        //        {
        //            x.Storage[x2.Item1, x2.Item2] = x2.Item3 * (1 - x2.Item3);
        //        }
        //        return x;
        //    }

        //    foreach (Tuple<int, int, double> x2 in x.Storage.EnumerateIndexed())
        //    {
        //        x.Storage[x2.Item1, x2.Item2] = 1 / (1 + Math.Exp(-x2.Item3));
        //    }
        //    return x;
        //}

        //public static Matrix<double> binaryCrossEnetropy(Matrix<double> X, Matrix<double> Y, Matrix<double> sync)
        //{
        //    Matrix<double> J = Matrix<double>.Build.Random(15, 4);

        //    var m = X.ColumnCount * X.RowCount;

        //    foreach (Tuple<int, int, double> x in X.Storage.EnumerateIndexed())
        //    {
                
        //        var a = Y[x.Item1, x.Item2] * Matrix.Log(nonlin(X*sync));
        //        var b = (1 - Y[x.Item1, x.Item2]) * Matrix.Log(1 - nonlin(X * sync));
        //        J += a + b;

        //        //var a = y.Item3 * Matrix.Log(nonlin(X[y.Item1, y.Item2] * sync));
        //        //var b = Matrix.Log(1 - nonlin(X[y.Item1, y.Item2] * sync));
        //        //J += -(1 / m) * (a + (1 - y.Item3) * b);
        //    }

        //    J = -(1 / m) * J;

        //    return J;
        //}


        public static Matrix<double> GeneratePopulation(int p, int WSize)
        {
            Random r = new Random(1);

            double[,] population = new double[p,WSize+1];

            for (var i = 0; i < p; i++)
            {
                double[] model = new double[WSize+1];

                for (var j = 0; j < (WSize + 1); j++)
                {
                    model[j] = (2 * r.NextDouble() - 1);
                    population[i, j] = model[j];
                }
            }

            return DenseMatrix.OfArray(population);
        }

        public static List<double> Mutation(Matrix<double> genom, double t = 0.5, double m = 0.1)
        {
            Random r = new Random(1);

            List<double> mutant = new List<double>();

            foreach (Tuple<int,int,double> gen in genom.Storage.EnumerateIndexed())
            {
                if (r.NextDouble() <= t)
                {
                    genom.Storage[gen.Item1,gen.Item2] += m * (2 * r.NextDouble() - 1);
                }

                mutant.Add(genom.Storage[gen.Item1, gen.Item2]);
            }

            return mutant;
        }

        static void Main(string[] args)
        {
            double[][] inputs =
            {
                new double[] {0.290221,0.129412,0.2},
                new double[] {0.290221,0.776471,0},
                new double[] {0.195584,0.070588,0.010204},
                new double[] {0.028391,0.247059,0.030612},
                new double[] {0.085174, 0.094118, 0.591837},
                new double[] {0.195584, 0.541176, 0.010204},
                new double[] {0.132492, 0.776471, 0.183673},
                new double[] {0.305994, 0.188235, 0.642857},
                new double[] {0.012618, 0.047059, 0.081633},
                new double[] {0, 0, 0.030612},
                new double[] {0.037855, 0.094118, 0.040816},
                new double[] {1, 1, 0.183673},
                new double[] {0.148265, 0.188235,1},
                new double[] {0.369085, 0.423529, 0.183673},
                new double[] {0.684543, 0.658824, 0.183673}
            };

            int[] outputs = // those are the class labels
            {
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
            };

            // Create a one-vs-one multi-class SVM learning algorithm 
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Learn a machine
            var machine = teacher.Learn(inputs, outputs);

            // Obtain class predictions for each sample
            int[] predicted = machine.Decide(inputs);

            // Compute classification accuracy
            double acc = new GeneralConfusionMatrix(expected: outputs, predicted: predicted).Accuracy;

            var bac = 0;



            Matrix<double> x = DenseMatrix.OfArray(new double[,] {
                                    {95, 15, 20},
                                    {95, 70, 2},
                                    {65, 10, 3},
                                    {12, 25, 5},
                                    {30, 12, 60},
                                    {65, 50, 3},
                                    {45, 70, 20},
                                    {100, 20, 65},
                                    {7, 8, 10},
                                    {3, 4, 5},
                                    {15, 12, 6},
                                    {320, 89, 20},
                                    {50, 20, 100},
                                    {120, 40, 20},
                                    {220, 60, 20} });

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

            x = Normalize(x);

            var a = GeneratePopulation(3, 5);
            var b = 0;

            //for (var i = 0; i < 100000; i++)
            //{
            //    var l0 = x;
                
            //    l1 = nonlin(l0 * syn0);
            //    l2 = nonlin(l1 * syn1);

            //    var l2_error = y - l2;

            //    if (i % 10000 == 0) 
            //        Console.WriteLine( "Error: {0}", Statistics.Mean(l2_error.Storage.Enumerate()));

            //    var refer2 = nonlin(l1 * syn1);
            //    // Метод наименьших квадратов (квадратичная ошибка)
            //    //var l2_delta = l2_error.PointwiseMultiply( binaryCrossEnetropy(l1,y,syn1));
            //    var l2_delta = l2_error.PointwiseMultiply(nonlin(refer2, true));

            //    var l1_error = l2_delta * syn1.Transpose();
            //    var refer1 = nonlin(l0 * syn0);
            //    // Метод наименьших квадратов (квадратичная ошибка)
            //    //var l1_delta = l1_error.PointwiseMultiply(binaryCrossEnetropy(l0, y, syn0));
            //    var l1_delta = l1_error.PointwiseMultiply(nonlin(refer1, true));

            //    syn1 += l1.Transpose() * l2_delta;
            //    syn0 += l0.Transpose() * l1_delta;

            //}

            //Console.WriteLine("Выходные данные после тренировки:");
            //Console.WriteLine(l1);

            Console.WriteLine("Выходные данные после тренировки:");
            Console.WriteLine(l2.PointwiseRound());


            Matrix<double> x_test = DenseVector.OfArray(new double[] { 36.56, 54.87, 15.49}).ToColumnMatrix();
            x_test = Normalize(x_test);
            
            //l1 = nonlin(x_test.Transpose() * syn0);
            //l2 = nonlin(l1 * syn1);

            Console.WriteLine("Ответ по тестовой выборке:");
            Console.WriteLine(l2);

            Console.ReadKey();
        }
    }
}
