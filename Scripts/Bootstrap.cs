using System;

class Program
{
    static void Main(string[] args)
    {
        TestNetwork.Run();
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

        input.PrintShape();
        target.PrintShape();
        logits.PrintShape();
        prediction.PrintShape();
        Console.WriteLine("Prediction:");
        PrintMatrix(prediction.Transpose());
        PrintMatrix(target);
        float loss = MathsUtils.CrossEntropyLoss(prediction, target);
        Matrix grad = MathsUtils.CrossEntropyGradient(prediction, target);

        Console.WriteLine($"\nLoss: {loss}");
        network.Backward(grad);

        Console.WriteLine("\nBackpropagation complete.");
    }

    private static Matrix GenerateInput(int size)
    {
        float[] data = new float[size];
        Random rand = new Random();
        for (int i = 0; i < size; i++)
            data[i] = (float)(rand.NextDouble() * 2 - 1);
        return new Matrix(data).Transpose(); // shape (size, 1)
    }

    private static Matrix GenerateOneHotTarget(int size)
    {
        float[] data = new float[size];
        Random rand = new Random();
        int hotIndex = rand.Next(size);
        data[hotIndex] = 1f;
        return new Matrix(data).Transpose(); // shape (size, 1)
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