using System;

class Program
{
    static void Main(string[] args)
    {

    }
}

public static class ModelInstanceTest
{

    public static void RunTest()
    {
        int inputDim = 3;
        int headDim = 16;
        int maxSeqLen = 12;
        int[] hiddenTopology = { 3, 5 };
        int outputDim = 4;
        ModelInstance instance = new ModelInstance(inputDim, headDim, maxSeqLen, hiddenTopology, outputDim);


    }
}
