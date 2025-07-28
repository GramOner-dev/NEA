using System;

class SelfAttention
{
    private AttentionProjection QueryProjection;
    private AttentionProjection KeyProjection;
    private AttentionProjection ValueProjection;

    public SelfAttention(int inputDim, int headDim, int maxSeqLen)
    {
        QueryProjection = new AttentionProjection(inputDim, headDim);
        KeyProjection = new AttentionProjection(inputDim, headDim);
        ValueProjection = new AttentionProjection(inputDim, headDim);
    }
}

class AttentionProjection
{
    private Matrix Weights;
    private Matrix WeightGradients;
    private AdamW WeightOptimizer;

    private Matrix Bias;
    private Matrix BiasGradients;
    private AdamW BiasOptimizer;

    public AttentionProjection(int inputDim, int headDim)
    {
        this.Weights = new Matrix(inputDim, headDim);
        this.WeightGradients = new Matrix(inputDim, headDim);
        this.WeightOptimizer = new AdamW(Weights, WeightGradients);

        this.Bias = new Matrix(headDim);
        this.BiasGradients = new Matrix(headDim);
        this.BiasOptimizer = new AdamW(Bias, BiasGradients);
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