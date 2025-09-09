using System;

class Program
{
    static void Main(string[] args)
    {
        AttentionTest.Run();
    }
}

public static class AttentionTest
{

    public static void Run()
    {
        int inputDim = 3;
        int maxSeqLen = 4;
        int headDim = 5;
        SelfAttention Transformer = new SelfAttention(inputDim, headDim, maxSeqLen);
        float[,] input = new float[,] { { 3, 4, 1 }, { 3, 4, 1 } };
        Matrix input = new Matrix();
        Matrix correctOutputs = new Matrix([0f, 0f, 1f]);
        int epochs = 20;
        for (int i = 0; i < epochs; i++)
        {
            Matrix output = Transformer.Forward(input);
        }
    }
}

public static class ModelInstanceTest
{

    public static void Run()
    {
        int inputDim = 32;
        int headDim = 16;
        int maxSeqLen = 12;
        int[] hiddenTopology = { 10, 5 };
        int outputDim = 3;
        int epochs = 200;
        ModelInstance instance = new ModelInstance(inputDim, headDim, maxSeqLen, hiddenTopology, outputDim);
        for (int i = 0; i < epochs; i++)
        {
            Matrix input = GenerateInput(32);
            Matrix target = GenerateOneHotTarget(3);

            Matrix logits = instance.Forward(input);
            Matrix.PrintMatrix(logits);
            Matrix prediction = MathsUtils.Softmax(logits);
            Console.WriteLine("Prediction - ");
            PrintMatrix(prediction.Transpose());

            float loss = MathsUtils.CrossEntropyLoss(prediction.Transpose(), target);
            Matrix grad = MathsUtils.CrossEntropyGradient(prediction.Transpose(), target);

            Console.WriteLine($"\nLoss - {loss}");
            instance.Backward(grad);
        }

    }
    private static Matrix GenerateInput(int size)
    {
        float[] data = new float[size];
        Random rand = new Random();
        for (int i = 0; i < size; i++)
            data[i] = (float)(rand.NextDouble() * 2 - 1);
        return new Matrix(data).Transpose();
    }

    private static Matrix GenerateOneHotTarget(int size)
    {
        float[] data = new float[size];
        Random rand = new Random();
        int hotIndex = rand.Next(size);
        data[0] = 1f;
        return new Matrix(data).Transpose();
    }
    private static void PrintMatrix(Matrix matrix)
    {
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                Console.Write($"{matrix[i, j]:0.0000} ");
            }
            Console.WriteLine();
        }
    }

}
