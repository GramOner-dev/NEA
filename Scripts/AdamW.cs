using System;


public class AdamW {
    private float beta1;
    private float beta2;
    private float epsilon;
    private int timestep;
    private float learningRate;
    private float decayRate;
    private Matrix firstMoments;
    private Matrix secondMoments;
    
    public AdamW(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float decayRate = 0.01f) {
        learningRate = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.decayRate = decayRate;
        timestep = 0;
    }

    private void Initialize(int rows, int cols){
        firstMoments = new Matrix(rows, cols);
        secondMoments = new Matrix(rows, cols);
    }

    public float getUnbiasedFirstMoment(float moment) => 
        moment / (1 - (float)Math.Pow(beta1, t));

    public float getUnbiasedSecondMoment(float moment) =>
        moment / (1 - (float)Math.Pow(beta2, t));

    public float getUpdatedFirstMoment(float moment, float grad) =>
        beta1 * moment + (1 - beta1) * grad;

    public float getUpdatedSecondMoment(float moment, float grad) =>
        beta2 * moment + (1 - beta2) * grad * grad;


    public void Update(Matrix parameters, Matrix grads) {
        if (firstMoments == null || secondMoments == null || firstMoments.Length != parameters.Length)
            Initialize(parameters.GetLength(0), parameters.GetLength(1));

        t++;

        for (int i = 0; i < parameters.GetLength(0); i++) {
            for (int j = 0; j < parameters.GetLength(1); j++) {
            
                firstMoments[i, j] = getUpdatedFirstMoment(firstMoments[i, j]);
                secondMoments[i, j] = getUpdatedSecondMoment(secondMoments[i, j]);

                float unbiasedFirstMoment = getUnbiasedFirstMoment(firstMoments[i, j], grads[i, j]);
                float unbiasedSecondMoment = getUnbiasedSecondMoment(secondMoments[i, j], grads[i, j]);

                parameters[i,j] -= learningRate * (unbiasedFirstMoment / (MathF.Sqrt(unbiasedSecondMoment) + epsilon) + weightDecay * parameters[i,j]);
            }
        }
    }
}