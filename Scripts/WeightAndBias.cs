public enum InitType
{
    Xavier,
    He
}

class WeightBiasPair
{
    private Matrix Weights;
    private Matrix WeightGradients;
    private AdamW WeightOptimizer;

    private Matrix Bias;
    private Matrix BiasGradients;
    private AdamW BiasOptimizer;

    public WeightBiasPair(int inputDim, int outputDim, InitType initType = InitType.He)
    {
        Weights = new Matrix(inputDim, outputDim);

        switch (initType)
        {
            case InitType.Xavier:
                {
                    Matrix XavierInitWeights = MathsUtils.XavierInit(inputDim, outputDim);
                    Weights.SetMatrix(XavierInitWeights);
                    break;
                }
            case InitType.He:
                {
                    Matrix HeInitWeights = MathsUtils.HeInit(inputDim, outputDim);
                    Weights.SetMatrix(HeInitWeights);
                    break;
                }
        }
        WeightGradients = new Matrix(inputDim, outputDim);
        WeightOptimizer = new AdamW(Weights, WeightGradients);

        Bias = new Matrix(outputDim);
        BiasGradients = new Matrix(outputDim);
        BiasOptimizer = new AdamW(Bias, BiasGradients);
    }

    public Matrix Forward(Matrix input)
    {
        Matrix BroadcastedBias = Matrix.BroadcastColumn(Bias, input.GetLength(1));
        return Weights.Transpose() * input + BroadcastedBias;
    }
    public void Update()
    {
        WeightOptimizer.Update();
        BiasOptimizer.Update();
    }
    public void SetWeightGradients(Matrix gradients) => WeightGradients.SetMatrix(gradients);
    public void SetBiasGradients(Matrix gradients) => BiasGradients.SetMatrix(gradients);
    public Matrix GetWeights() => Weights;

}