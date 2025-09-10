public class SelfAttention
{
    private WeightBiasPair QueryProjection;
    private WeightBiasPair KeyProjection;
    private WeightBiasPair ValueProjection;
    private LayerNormalizer LayerNorm;
    private PositionalEncoding PosEnc;

    private Matrix input;
    private Matrix queries;
    private Matrix keys;
    private Matrix values;
    private Matrix attentionScores;
    private Matrix attentionWeights;
    private Matrix context;
    private Matrix normed;

    public SelfAttention(int inputDim, int headDim, int maxSeqLen)
    {
        QueryProjection = new WeightBiasPair(inputDim, headDim);
        KeyProjection = new WeightBiasPair(inputDim, headDim);
        ValueProjection = new WeightBiasPair(inputDim, headDim);
        LayerNorm = new LayerNormalizer(headDim);
        PosEnc = new PositionalEncoding(inputDim, maxSeqLen);

        //temp
        input = new Matrix();
        queries = new Matrix();
        keys = new Matrix();
        values = new Matrix();
        attentionScores = new Matrix();
        attentionWeights = new Matrix();
        context = new Matrix();
        normed = new Matrix();

    }


    public Matrix Forward(Matrix input)
    {
        this.input = PosEnc.Forward(input);

        queries = QueryProjection.Forward(this.input);
        keys = KeyProjection.Forward(this.input);
        values = ValueProjection.Forward(this.input);

        attentionScores = ComputeAttentionScores(queries, keys);
        attentionWeights = MathsUtils.Softmax(attentionScores);
        context = attentionWeights.Transpose() * values;

        normed = LayerNorm.Forward(context.Transpose());

        // Matrix.PrintMatrix(normed);
        return MathsUtils.MeanPool(normed.Transpose());
    }

    public void Backward(Matrix dLdOutput)
    {
        Matrix dLdNormed = MathsUtils.ExpandGradThroughMeanPool(dLdOutput, normed.GetLength(0)).Transpose();
        Matrix dLdContext = LayerNorm.Backward(dLdNormed).Transpose();

        Matrix dLdValues = attentionWeights.Transpose() * dLdContext;
        Matrix dLdAttentionWeights = dLdContext * values.Transpose();
        Matrix dLdAttentionScores = MathsUtils.BackpropSoftmax(dLdAttentionWeights, attentionWeights);
        Matrix dLdQueries = dLdAttentionScores.Transpose() * keys;
        Matrix dLdKeys = dLdAttentionScores.Transpose() * queries;

        Matrix QueryWeightGrads = dLdQueries * input.Transpose();
        Matrix QueryBiasGrads = dLdQueries.RowWiseSum();
        QueryProjection.SetWeightGradients(QueryWeightGrads.Transpose());
        QueryProjection.SetBiasGradients(QueryBiasGrads);

        Matrix KeyWeightGrads = dLdKeys * input.Transpose();
        Matrix KeyBiasGrads = dLdKeys.RowWiseSum();
        KeyProjection.SetWeightGradients(KeyWeightGrads.Transpose());
        KeyProjection.SetBiasGradients(KeyBiasGrads);

        Matrix ValueWeightGrads = dLdValues * input.Transpose();
        Matrix ValueBiasGrads = dLdValues.RowWiseSum();
        ValueProjection.SetWeightGradients(ValueWeightGrads.Transpose());
        ValueProjection.SetBiasGradients(ValueBiasGrads);
        UpdateWeights();
    }

    public void UpdateWeights()
    {
        QueryProjection.Update();
        KeyProjection.Update();
        ValueProjection.Update();
        LayerNorm.UpdateWeights();
        PosEnc.Update();
    }

    private Matrix ComputeAttentionScores(Matrix Query, Matrix Key)
    {
        Matrix KeyT = Key.Transpose();
        float dotProdScaling = (float)Math.Sqrt(Query.GetLength(0));
        return (Query * KeyT).Apply(x => x / dotProdScaling);
    }
}