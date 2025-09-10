
using System.Runtime.InteropServices;

class PositionalEncoding
{
    private Matrix Weights;
    private Matrix WeightGradients;
    private AdamW optimizer;

    private int inputDim;
    private Matrix usedEncodings;

    public PositionalEncoding(int inputDim, int maxSeqLen)
    {
        this.inputDim = inputDim;
        this.usedEncodings = new Matrix(1, 1); //temp

        Weights = new Matrix(inputDim, maxSeqLen);
        WeightGradients = new Matrix(inputDim, maxSeqLen);
        optimizer = new AdamW(Weights, WeightGradients);

        Weights.Fill(0.01f);
    }

    public Matrix Forward(Matrix input)
    {
        int seqLen = input.GetLength(0);

        float[,] encodingValues = new float[inputDim, seqLen];

        for (int i = 0; i < inputDim; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                encodingValues[i, j] = Weights[i, j];
            }
        }

        usedEncodings = new Matrix(encodingValues);
        Matrix output = input.Transpose() + usedEncodings;
        return output; //inputDim(rows) x seqLen(cols)
    }

    public Matrix Backward(Matrix gradOutput)
    {
        WeightGradients.Fill(0f);
        for (int i = 0; i < usedEncodings.GetLength(1); i++)
        {
            for (int j = 0; j < inputDim; j++)
            {
                WeightGradients[j, i] += gradOutput[j, i];
            }
        }

        return gradOutput;
    }

    public void Update() => optimizer.Update();
}