public class ModelInstance
{
    #region TransformerConfig
    public SelfAttention Transformer;
    public int inputDim;
    public int maxSeqLen;
    #endregion

    #region NNConfig
    public Network NNetwork;
    #endregion

    //maybe ModelConfig class later, allows to change through reference during runtime instead of remaking model
    //might make reshaping the weights and things difficult so tbd.
    //will need to make sure garbage collection 100% works as otherwise memory leak might occur from millions of weights
    public ModelInstance(int inputDim, int headDim, int maxSeqLen, int[] hiddenLayersTopology, int outputDim)
    {
        this.inputDim = inputDim;
        this.maxSeqLen = maxSeqLen;
        Transformer = new SelfAttention(inputDim, headDim, maxSeqLen);

        NNetwork = new Network(headDim, hiddenLayersTopology, outputDim);
    }

    public Matrix Forward(Matrix input)
    {
        //Matrix.PrintMatrix(input.Transpose());
        Matrix output = Transformer.Forward(input);
        Console.WriteLine("attention output");
        Matrix.PrintMatrix(output);
        output = NNetwork.Forward(output);
        //Matrix.PrintMatrix(output.Transpose());

        return output;
    }

    public void Backward(Matrix dLdOutput)
    {
        Matrix dLdNNInput = NNetwork.Backward(dLdOutput);
        Transformer.Backward(dLdNNInput);
    }


}