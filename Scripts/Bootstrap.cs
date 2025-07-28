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
        Matrix target = GenerateTarget(3);

        // Forward pass
        Matrix prediction = network.Forward(input);  // assume you make forward() public or use wrapper
        Console.WriteLine("Prediction:");
        PrintMatrix(prediction);

        // Compute MSE and its gradient
        float loss = MathsUtils.MSE(prediction, target);
        Matrix grad = MathsUtils.MSEGradient(prediction, target);

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

    private static Matrix GenerateTarget(int size)
    {
        float[] data = new float[size];
        Random rand = new Random();
        for (int i = 0; i < size; i++)
            data[i] = (float)(rand.NextDouble()); // values in [0, 1]
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