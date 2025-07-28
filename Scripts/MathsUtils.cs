using System;

public static class MathsUtils {
    private static Random random = new Random();

    public static float NextGaussian(double mean = 0, double stddev = 1){   
        double epsilon = 1e-10;
        double r1 = random.NextDouble() + epsilon; 
        double r2 = random.NextDouble() + epsilon;
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(r1)) *  Math.Sin(2.0 * Math.PI * r2); 
        return (float)(mean + stddev * randStdNormal);
    }

    public static float[] HeInit(int numOfInputs){
        float[] weights = new float[numOfInputs];
        float stdDev = (float)Math.Sqrt(2.0 / numOfInputs);
        for(int i = 0; i < numOfInputs; i++)
            weights[i] = NextGaussian(0, stdDev);
        
        return weights;
    }

    public static Matrix Softmax(Matrix input) {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            float[] row = input[i];

            float max = row.Max();

            float[] exps = new float[cols];
            float sum = 0f;

            for (int j = 0; j < cols; j++) {
                exps[j] = (float)Math.Exp(row[j] - max);
                sum += exps[j];
            }

            for (int j = 0; j < cols; j++) {
                result[i, j] = exps[j] / sum;
            }
        }

        return result;
    }

    private static float LeakyReLUnegativeGradient = 0.001f;

    public static float LeakyReLU(float value) => 
        value > 0 ? value : LeakyReLUnegativeGradient * value;

    public static float LeakyReLUDeriv(float value) =>
        value > 0 ? 1 : LeakyReLUnegativeGradient;
}

