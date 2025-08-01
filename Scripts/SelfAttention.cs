using System;

class SelfAttention
{
    private WeightBiasPair QueryProjection;
    private WeightBiasPair KeyProjection;
    private WeightBiasPair ValueProjection;
    private LayerNormalizer LayerNorm;
    private PositionalEncoding PosEnc;
    public SelfAttention(int inputDim, int headDim, int maxSeqLen)
    {
        QueryProjection = new WeightBiasPair(inputDim, headDim);
        KeyProjection = new WeightBiasPair(inputDim, headDim);
        ValueProjection = new WeightBiasPair(inputDim, headDim);
        LayerNorm = new LayerNormalizer(headDim);
        PosEnc = new PositionalEncoding(inputDim, maxSeqLen);
    }
}



class PositionalEncoding
{
    private Matrix Weights;
    private Matrix WeightGradients;
    private AdamW optimizer;
    public PositionalEncoding(int inputDim, int maxSeqLen)
    {
        Weights = new Matrix(inputDim, maxSeqLen);
        WeightGradients = new Matrix(inputDim, maxSeqLen);
        optimizer = new AdamW(Weights, WeightGradients);
    }
}