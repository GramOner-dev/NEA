
public static class TestNetwork
{
    public static void Run()
    {
        int numOfInputs = 3;
        int numOfOutputs = 3;
        int[] hiddenLayersTopology = { 5, 5 };
        Network network = new Network(numOfInputs, hiddenLayersTopology, numOfOutputs);
        int epochs = 200;
        Matrix input = new Matrix([3, 4, 1]);
        Matrix correctOutputs = new Matrix([0f, 0f, 1f]);



        for (int i = 0; i < epochs; i++)
        {
            Matrix logits = network.Forward(input.Transpose());
            Matrix prediction = MathsUtils.Softmax(logits);
            Console.WriteLine("Prediction:");
            PrintMatrix(prediction.Transpose());
            Console.WriteLine("inputs:");
            PrintMatrix(input.Transpose());

            float loss = MathsUtils.CrossEntropyLoss(prediction, correctOutputs);
            Matrix grad = MathsUtils.CrossEntropyGradient(prediction, correctOutputs);

            Console.WriteLine($"\nLoss: {loss}");
            network.Backward(grad);
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
        data[hotIndex] = 1f;
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

public static class LayerNormTest
{
    public static void Run()
    {
        Console.WriteLine("LayerNormTest");

        int seqLen = 1;
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

        Console.WriteLine("input-");
        PrintMatrix(input);
        Matrix gradOutput = new Matrix(seqLen, hiddenDim);
        gradOutput = gradOutput.Apply(x => (float)rand.NextDouble());

        for (int i = 0; i < 100; i++)
        {
            Matrix output = norm.Forward(input);
            Console.WriteLine("\noutput after LayerNorm:");
            PrintMatrix(output);




            Matrix gradInput = norm.Backward(gradOutput);
            Console.WriteLine("dLdInput-:");
            PrintMatrix(gradInput);
            Console.WriteLine("dLdOutput-:");
            PrintMatrix(gradOutput);


            norm.UpdateWeights();

            Console.WriteLine("\ngamma after update");
            PrintMatrix(norm.Gamma);
            Console.WriteLine("\nbeta after update");
            PrintMatrix(norm.Beta);
        }


    }

    private static void PrintMatrix(Matrix matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            Console.Write("[ ");
            for (int j = 0; j < cols; j++)
            {
                Console.Write($"{matrix[i, j]:0.00} ");
            }
            Console.WriteLine("]");
        }
    }
}