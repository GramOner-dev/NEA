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
        // Define network topology: 3 inputs, 2 hidden, 1 output
        int[] topology = { 32, 5, 3 };
        Network network = new Network(topology);

        // Generate a single random input and target
        Matrix input = GenerateInput(32);
        Matrix target = GenerateOneHotTarget(3);

        // Forward pass
        Matrix logits = network.Forward(input);
        Matrix prediction = MathsUtils.Softmax(logits);
        Console.WriteLine("Prediction:");
        PrintMatrix(prediction);

        // Compute MSE and its gradient
        float loss = MathsUtils.CrossEntropyLoss(logits, target);
        Matrix grad = MathsUtils.CrossEntropyGradient(logits, target);

        Console.WriteLine($"\nLoss: {loss}");

        // Backward pass
        network.Backward(grad);  // assume you make backward() public or use wrapper

        Console.WriteLine("\nBackpropagation complete.");
    }

    private static Matrix GenerateInput(int size)
    {
        float[] data = new float[size];
        Random rand = new Random();
        for (int i = 0; i < size; i++)
            data[i] = (float)(rand.NextDouble() * 2 - 1); // values in [-1, 1]
        return new Matrix(data).Transpose(); // shape (size, 1)
    }

    private static Matrix GenerateOneHotTarget(int size)
    {
        float[] data = new float[size];
        Random rand = new Random();
        int hotIndex = rand.Next(size); // Choose one index to be hot
        data[hotIndex] = 1f;            // One-hot encoding
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