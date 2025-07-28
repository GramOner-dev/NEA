using System;

class Program
{
    static void Main(string[] args)
    {
        LayerNormTest.Run();
    }
}

public static class TestNetwork


{
    public static void Run()
    {
        int[] topology = { 32, 5, 3 };
        Network network = new Network(topology);

        Matrix input = GenerateInput(32);
        Matrix target = GenerateOneHotTarget(3);

        Matrix logits = network.Forward(input);
        Matrix prediction = MathsUtils.Softmax(logits);
        Console.WriteLine("Prediction:");
        PrintMatrix(prediction.Transpose());

        float loss = MathsUtils.CrossEntropyLoss(prediction, target);
        Matrix grad = MathsUtils.CrossEntropyGradient(prediction, target);

        Console.WriteLine($"\nLoss: {loss}");
        network.Backward(grad);
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
        data[hotIndex] = 1f;
        return new Matrix(data).Transpose();
    }




    private static void PrintMatrix(Matrix m)
    {
        for (int i = 0; i < m.GetLength(0); i++)
        {
            for (int j = 0; j < m.GetLength(1); j++)
            {
                Console.Write($"{m[i, j]:0.0000} ");
            }
            Console.WriteLine();
        }
    }
}

public static class LayerNormTest
{
    public static void Run()
    {
        Console.WriteLine("Running LayerNormalizer test...");

        int seqLen = 4;
        int hiddenDim = 5;
        LayerNormalizer norm = new LayerNormalizer(hiddenDim);


        Matrix input = new Matrix(seqLen, hiddenDim);
        Random rand = new Random();

        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                input[i, j] = (float)(rand.NextDouble() * 10);
            }
        }

        Console.WriteLine("Input:");
        PrintMatrix(input);


        Matrix output = norm.Forward(input);
        Console.WriteLine("\nOutput after LayerNorm:");
        PrintMatrix(output);


        Random random = new Random();
        Matrix gradOutput = new Matrix(seqLen, hiddenDim);
        gradOutput = gradOutput.Apply(x => (float)random.NextDouble());


        Matrix gradInput = norm.Backward(gradOutput);
        Console.WriteLine("\nGradInput:");
        PrintMatrix(gradInput);


        norm.UpdateWeights();

        Console.WriteLine("\nGamma after update:");
        PrintMatrix(norm.Gamma);
        Console.WriteLine("\nBeta after update:");
        PrintMatrix(norm.Beta);

        Console.WriteLine("\nLayerNormalizer test complete.");
    }

    private static void PrintMatrix(Matrix m)
    {
        int rows = m.GetLength(0);
        int cols = m.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            Console.Write("[ ");
            for (int j = 0; j < cols; j++)
            {
                Console.Write($"{m[i, j]:0.00} ");
            }
            Console.WriteLine("]");
        }
    }
}