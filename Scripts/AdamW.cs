using System;


public class AdamW
{
    private float beta1;
    private float beta2;
    private float epsilon;
    private int timestep;
    private float learningRate;
    private float decayRate;
    private Matrix firstMoments;
    private Matrix secondMoments;
    private Matrix parameters;
    private Matrix gradients;
    public AdamW(Matrix parameters, Matrix gradients, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float decayRate = 0.01f)
    {
        learningRate = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.decayRate = decayRate;
        (int rows, int cols) = parameters.Shape();
        this.firstMoments = new Matrix(rows, cols);
        this.secondMoments = new Matrix(rows, cols);
        this.parameters = parameters;
        this.gradients = gradients;
        timestep = 0;

    }


    public float getUnbiasedFirstMoment(float moment) =>
        moment / (1 - (float)Math.Pow(beta1, timestep));

    public float getUnbiasedSecondMoment(float moment) =>
        moment / (1 - (float)Math.Pow(beta2, timestep));

    public float getUpdatedFirstMoment(float moment, float grad) =>
        beta1 * moment + (1 - beta1) * grad;

    public float getUpdatedSecondMoment(float moment, float grad) =>
        beta2 * moment + (1 - beta2) * grad * grad;


    public void Update()
    {
        timestep++;

        for (int i = 0; i < parameters.GetLength(0); i++)
        {
            for (int j = 0; j < parameters.GetLength(1); j++)
            {

                firstMoments[i, j] = getUpdatedFirstMoment(firstMoments[i, j], gradients[i, j]);
                secondMoments[i, j] = getUpdatedSecondMoment(secondMoments[i, j], gradients[i, j]);

                float unbiasedFirstMoment = getUnbiasedFirstMoment(firstMoments[i, j]);
                float unbiasedSecondMoment = getUnbiasedSecondMoment(secondMoments[i, j]);

                parameters[i, j] -= learningRate * (unbiasedFirstMoment / (MathF.Sqrt(unbiasedSecondMoment) + epsilon) + decayRate * parameters[i, j]);
            }
        }
    }
}