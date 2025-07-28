public class LayerNormalizer
{
    private float epsilon = 1e-5f;

    public Matrix Gamma;
    public Matrix GammaGrads;
    public AdamW GammaOptimizer;
    public Matrix Beta;
    public Matrix BetaGrads;
    public AdamW BetaOptimizer;


    private Matrix inputs;
    private float[] means;
    private float[] variances;
    private Matrix normalizedInputs;

    public LayerNormalizer(int inputDim)
    {

        inputs = new Matrix(inputDim, 1);
        means = Array.Empty<float>();
        variances = Array.Empty<float>();
        normalizedInputs = new Matrix(inputDim, 1);

        Gamma = new Matrix(1, inputDim);
        GammaGrads = new Matrix(1, inputDim);
        GammaOptimizer = new AdamW(Gamma, GammaGrads);

        Beta = new Matrix(1, inputDim);
        BetaGrads = new Matrix(1, inputDim);
        BetaOptimizer = new AdamW(Beta, BetaGrads);


        for (int i = 0; i < inputDim; i++)
        {
            Gamma[0, i] = 1f;
            Beta[0, i] = 0f;
        }
    }

    public Matrix Forward(Matrix input)
    {
        int seqLen = input.GetLength(0);
        int hiddenDim = input.GetLength(1);

        inputs = input;
        means = new float[seqLen];
        variances = new float[seqLen];
        normalizedInputs = new Matrix(seqLen, hiddenDim);
        Matrix output = new Matrix(seqLen, hiddenDim);

        for (int i = 0; i < seqLen; i++)
        {
            float[] row = input[i];

            float mean = CalculateMean(row);
            means[i] = mean;

            float variance = CalculateVariance(row, mean);
            variances[i] = variance;

            float[] normalizedRow = NormalizeRow(row, mean, variance);
            normalizedInputs[i] = normalizedRow;

            float[] outputRow = ApplyGammaBeta(normalizedRow);
            output[i] = outputRow;
        }

        return output;
    }

    public Matrix Backward(Matrix gradOutput)
    {
        int seqLen = gradOutput.GetLength(0);
        int hiddenDim = gradOutput.GetLength(1);

        Matrix gradInput = new Matrix(seqLen, hiddenDim);
        GammaGrads.Fill(0f);
        BetaGrads.Fill(0f);

        for (int i = 0; i < seqLen; i++)
        {
            float[] row = inputs[i];
            float[] gradOutRow = gradOutput[i];
            float[] normRow = normalizedInputs[i];
            float mean = means[i];
            float variance = variances[i];
            float stdDev = (float)Math.Sqrt(variance + epsilon);

            AccumulateGammaBetaGrads(gradOutRow, normRow, GammaGrads, BetaGrads);
            float[] gradNormalized = CalculateGradNormalized(gradOutRow);

            float gradVariance = CalculateGradVariance(row, mean, variance, gradNormalized);
            float gradMean = CalculateGradMean(row, mean, variance, gradNormalized, gradVariance);

            float[] gradInputRow = CalculateGradInputRow(row, mean, stdDev, gradNormalized, gradVariance, gradMean);
            gradInput[i] = gradInputRow;
        }

        return gradInput;
    }

    public void UpdateWeights()
    {
        GammaOptimizer.Update();
        BetaOptimizer.Update();
    }

    private float CalculateMean(float[] row)
    {
        float sum = 0f;
        for (int j = 0; j < row.Length; j++)
            sum += row[j];
        return sum / row.Length;
    }

    private float CalculateVariance(float[] row, float mean)
    {
        float sumSq = 0f;
        for (int j = 0; j < row.Length; j++)
        {
            float diff = row[j] - mean;
            sumSq += diff * diff;
        }
        return sumSq / row.Length;
    }

    private float[] NormalizeRow(float[] row, float mean, float variance)
    {
        int len = row.Length;
        float[] normalized = new float[len];
        float stdDev = (float)Math.Sqrt(variance + epsilon);

        for (int j = 0; j < len; j++)
            normalized[j] = (row[j] - mean) / stdDev;

        return normalized;
    }

    private float[] ApplyGammaBeta(float[] normalizedRow)
    {
        int len = normalizedRow.Length;
        float[] outputRow = new float[len];

        for (int j = 0; j < len; j++)
            outputRow[j] = Gamma[0, j] * normalizedRow[j] + Beta[0, j];

        return outputRow;
    }

    private void AccumulateGammaBetaGrads(float[] gradOutRow, float[] normRow, Matrix GammaGrads, Matrix BetaGrads)
    {
        for (int j = 0; j < gradOutRow.Length; j++)
        {
            GammaGrads[0, j] += gradOutRow[j] * normRow[j];
            BetaGrads[0, j] += gradOutRow[j];
        }
    }

    private float[] CalculateGradNormalized(float[] gradOutRow)
    {
        int len = gradOutRow.Length;
        float[] gradNormalized = new float[len];

        for (int j = 0; j < len; j++)
            gradNormalized[j] = gradOutRow[j] * Gamma[0, j];

        return gradNormalized;
    }

    private float CalculateGradVariance(float[] row, float mean, float variance, float[] gradNormalized)
    {
        float gradVariance = 0f;
        for (int j = 0; j < row.Length; j++)
        {
            gradVariance += gradNormalized[j] * (row[j] - mean) * -0.5f * (float)Math.Pow(variance + epsilon, -1.5f);
        }
        return gradVariance;
    }

    private float CalculateGradMean(float[] row, float mean, float variance, float[] gradNormalized, float gradVariance)
    {
        float stdDev = (float)Math.Sqrt(variance + epsilon);
        float gradMean = 0f;

        for (int j = 0; j < row.Length; j++)
            gradMean += gradNormalized[j] * -1f / stdDev;

        float sumDiff = 0f;
        for (int j = 0; j < row.Length; j++)
            sumDiff += row[j] - mean;

        gradMean += gradVariance * -2f * sumDiff / row.Length;

        return gradMean;
    }

    private float[] CalculateGradInputRow(float[] row, float mean, float stdDev, float[] gradNormalized, float gradVariance, float gradMean)
    {
        int len = row.Length;
        float[] gradInputRow = new float[len];

        for (int j = 0; j < len; j++)
        {
            gradInputRow[j] =
                gradNormalized[j] / stdDev +
                gradVariance * 2f * (row[j] - mean) / len +
                gradMean / len;
        }

        return gradInputRow;
    }
}
