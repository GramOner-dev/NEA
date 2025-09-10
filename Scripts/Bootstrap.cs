using System;

class Program
{
    static void Main(string[] args)
    {
        ModelInstanceTest.Run();
    }
}

public static class ModelInstanceTest
{

    public static void Run()
    {
        int inputDim = 3;
        int maxSeqLen = 4;
        int headDim = 5;
        SelfAttention Transformer = new SelfAttention(inputDim, headDim, maxSeqLen);

        int numOfInputs = 5;
        int numOfOutputs = 5;
        int[] hiddenLayersTopology = { 7, 7 };
        Network network = new Network(numOfInputs, hiddenLayersTopology, numOfOutputs);


        Matrix input = new Matrix([3, 4, 1]);
        Matrix correctOutputs = new Matrix([0f, 0f, 0f, 0f, 1f]);
        int epochs = 20;
        for (int i = 0; i < epochs; i++)
        {
            Matrix AttentionLogits = Transformer.Forward(input);

            Matrix NNlogits = network.Forward(AttentionLogits.Transpose());
            Matrix prediction = MathsUtils.Softmax(NNlogits);
            PrintOutputs(input, AttentionLogits, NNlogits, prediction, correctOutputs);

            float loss = MathsUtils.CrossEntropyLoss(prediction, correctOutputs);
            Matrix grad = MathsUtils.CrossEntropyGradient(prediction.Transpose(), correctOutputs.Transpose());
            Console.WriteLine($"\nLoss: {loss}");
            Matrix dLdNNInputs = network.Backward(grad);
            Transformer.Backward(dLdNNInputs);
        }

    }

    private static void PrintOutputs(Matrix input, Matrix AttentionLogits, Matrix NNlogits, Matrix prediction, Matrix correctOutputs)
    {
        Console.WriteLine("inputs:");
        Matrix.PrintMatrix(input);
        Console.WriteLine("Attention logits:");
        Matrix.PrintMatrix(AttentionLogits.Transpose());
        Console.WriteLine("Network logits:");
        Matrix.PrintMatrix(NNlogits);
        Console.WriteLine("Prediction:");
        Matrix.PrintMatrix(prediction);
        Console.WriteLine("correctOutput:");
        Matrix.PrintMatrix(correctOutputs);
    }

}
