using System;

public static class MathsUtils
{
    #region Initialization
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
    #endregion

    #region Softmax
    public static Matrix Softmax(Matrix input)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
                if (input[i, j] > max)
                    max = input[i, j];

            float sum = 0f;
            float[] exps = new float[cols];
            for (int j = 0; j < cols; j++)
            {
                exps[j] = (float)Math.Exp(input[i, j] - max);
                sum += exps[j];
            }

            for (int j = 0; j < cols; j++)
                result[i, j] = exps[j] / sum;
        }

        return result;
    }

    public static Matrix BackpropSoftmax(Matrix dLdAttnWeights, Matrix attnWeights)
    {
        Matrix grad = new Matrix(attnWeights.GetLength(0), attnWeights.GetLength(1));
        for (int i = 0; i < grad.GetLength(0); i++)
        {
            for (int j = 0; j < grad.GetLength(1); j++)
            {
                float soft = attnWeights[i, j];
                grad[i, j] = dLdAttnWeights[i, j] * soft * (1 - soft);
            }
        }
        return grad;
    }
    #endregion

    #region MeanPooling
    public static Matrix MeanPool(Matrix input)
    {
        int cols = input.GetLength(1);
        Matrix result = new Matrix(input.GetLength(0), 1);
        for (int i = 0; i < input.GetLength(0); i++)
        {
            float sum = 0f;
            for (int j = 0; j < cols; j++)
                sum += input[i, j];
            result[i, 0] = sum / cols;
        }
        return result;
    }


    public static Matrix ExpandGradThroughMeanPool(Matrix dMeanPooled, int seqLen)
    {
        Matrix expanded = new Matrix(dMeanPooled.GetLength(0), seqLen);
        for (int i = 0; i < expanded.GetLength(0); i++)
            for (int j = 0; j < seqLen; j++)
                expanded[i, j] = dMeanPooled[i, 0] / seqLen;

        return expanded;
    }
    #endregion

    #region CrossEntropy
    public static float CrossEntropyLoss(Matrix probs, Matrix oneHotTarget)
    {
        int targetIndex = Array.IndexOf(oneHotTarget[0], 1f);
        float predictedProb = probs[0, targetIndex];
        return -MathF.Log(predictedProb + 1e-9f);
    }

    public static Matrix CrossEntropyGradient(Matrix probs, Matrix oneHotTarget)
    {
        probs = probs.Transpose();
        for (int i = 0; i < probs.GetLength(0); i++)
            probs[i, 0] -= oneHotTarget[i, 0];

        return probs.Transpose();
    }
    #endregion

    #region ReLU
    private static float LeakyReLUnegativeGradient = 0.001f;

    public static float LeakyReLU(float value) =>
        value > 0 ? value : LeakyReLUnegativeGradient * value;

    public static float LeakyReLUDeriv(float value) =>
        value > 0 ? 1 : LeakyReLUnegativeGradient;
    #endregion
}

