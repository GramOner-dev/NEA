using System;

public static class MathsUtils
{
    private static Random random = new Random();

    public static float NextGaussian(double mean = 0, double stddev = 1)
    {
        double epsilon = 1e-10;
        double r1 = random.NextDouble() + epsilon;
        double r2 = random.NextDouble() + epsilon;
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(r1)) * Math.Sin(2.0 * Math.PI * r2);
        return (float)(mean + stddev * randStdNormal);
    }

    public static Matrix HeInit(int rowNum, int colNum)
    {
        var matrix = new Matrix(rowNum, colNum);
        float stdDev = (float)Math.Sqrt(2.0 / colNum);

        for (int i = 0; i < rowNum; i++)
        {
            float[] row = new float[colNum];
            for (int j = 0; j < colNum; j++)
                row[j] = NextGaussian(0, stdDev);

            for (int j = 0; j < colNum; j++)
                matrix[i, j] = row[j];
        }

        return matrix;
    }

    public static Matrix XavierInit(int rowNum, int colNum)
    {
        var matrix = new Matrix(rowNum, colNum);
        float limit = (float)Math.Sqrt(6.0 / (rowNum + colNum));

        var rand = new Random();
        for (int i = 0; i < rowNum; i++)
        {
            float[] row = new float[colNum];
            for (int j = 0; j < colNum; j++)
                row[j] = (float)(rand.NextDouble() * 2 * limit - limit);

            for (int j = 0; j < colNum; j++)
                matrix[i, j] = row[j];
        }

        return matrix;
    }
    public static Matrix Softmax(Matrix input)
    {
        int rows = input.GetLength(0);

        Matrix result = new Matrix(rows, 1);
        float max = float.NegativeInfinity;

        for (int i = 0; i < rows; i++)
            if (input[i, 0] > max)
                max = input[i, 0];

        float sum = 0f;
        float[] exps = new float[rows];

        for (int i = 0; i < rows; i++)
        {
            exps[i] = (float)Math.Exp(input[i, 0] - max);
            sum += exps[i];
        }

        for (int i = 0; i < rows; i++)
            result[i, 0] = exps[i] / sum;

        Console.WriteLine("softmax:");
        result.PrintShape();
        return result;
    }

    public static float CrossEntropyLoss(Matrix probs, Matrix oneHotTarget)
    {
        oneHotTarget = oneHotTarget.Transpose();
        int targetIndex = Array.IndexOf(oneHotTarget[0], 1f);
        float predictedProb = probs[targetIndex, 0];
        float loss = -MathF.Log(predictedProb + 1e-9f);
        return loss;
    }

    public static Matrix CrossEntropyGradient(Matrix probs, Matrix oneHotTarget)
    {

        for (int i = 0; i < probs.GetLength(0); i++)
            probs[i, 0] -= oneHotTarget[i, 0];

        return probs;
    }

    private static float LeakyReLUnegativeGradient = 0.001f;

    public static float LeakyReLU(float value) =>
        value > 0 ? value : LeakyReLUnegativeGradient * value;

    public static float LeakyReLUDeriv(float value) =>
        value > 0 ? 1 : LeakyReLUnegativeGradient;
}

