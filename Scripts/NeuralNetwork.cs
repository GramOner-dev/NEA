using System;

public class Network{
    private Layer[] layers;
    private Matrix inputGradients;

    public Network(int[] topology){
        layers = new Layer[topology.Length - 1];

        for(int i = 1; i < topology.Length; i++){
            bool isOutputLayer = i == topology.Length - 1;
            layers[i - 1] = new Layer(topology[i - 1], topology[i], isOutputLayer);
        }
    }

    private Matrix forward(Matrix input){
        Matrix currentOutput = input;

        for(int i = 0; i < layers.Length; i++)
            currentOutput = layers[i].forward(currentOutput);
            
        return currentOutput;
    }

    private Matrix recursiveBackprop(int layerIndex, Matrix dLdOutput) {
        if (layerIndex < 0)
            return dLdOutput; 

        Matrix dLdInput = layers[layerIndex].backward(dLdOutput);
        layers[layerIndex].UpdateWeights();
        return recursiveBackprop(layerIndex - 1, dLdInput);
    }

    private Matrix backward(Matrix dLdOutput) {
        inputGradients = recursiveBackprop(layers.Length - 1, dLdOutput);
        return inputGradients;
    }
}

public class Layer{
    private Matrix weights;
    private Matrix weightGradients;
    private Matrix biases;
    private Matrix biasGradients;
    private Matrix input, logits, output;
    private bool isOutputLayer;
    private AdamW optimizer;

    public Matrix GetWeights() => weights;
    public Matrix SetWeights(Matrix weights) => this.weights = weights;
    
    public Matrix GetWeightGradients() => weightGradients;
    public Matrix SetWeightGradients(Matrix weightGradients) => this.weightGradients = weightGradients;

    public Matrix GetBiases() => biases;
    public Matrix SetBiases(Matrix biases) => this.biases = biases;
    
    public Matrix GetBiasGradients() => biasGradients;
    public Matrix SetBiasGradients(Matrix biasGradients) => this.biasGradients = biasGradients;

    public Layer(int numOfInputs, int numOfNeurons, bool isOutputLayer){
        weights = new Matrix(numOfNeurons, numOfInputs).HeInit();
        weightGradients = new Matrix(numOfNeurons, numOfInputs);

        biases = new Matrix(numOfNeurons);
        biasGradients = new Matrix(numOfNeurons);
        this.isOutputLayer = isOutputLayer;

        optimizer = new AdamW();
    }

    public Matrix forward(Matrix input){
        this.input = input;
        this.logits = (weights * input) + biases.Transpose();
        if(!isOutputLayer)
            this.output = logits.Apply(x => MathsUtils.LeakyReLU(x));
        else
            this.output = logits;
        
        return output;
    }

    private Matrix backward(Matrix nextLayerGradients){
        Matrix dLdPreActivation = nextLayerGradients.Hadamard(logits.Apply(MathsUtils.LeakyReLUDeriv()));

        if(!isOutputLayer)
            Matrix dLdPreActivation = nextLayerGradients;

        Matrix dLdWeights = dLdPreActivation * input.Transpose();
        this.weightGradients = dLdWeights;
        this.biasGradients = dLdPreActivation;

        Matrix dLdInput = weights.Transpose() * dLdPreActivation;
        return dLdInput;
    }

    private void UpdateWeights() => optimizer.Update(weights, weightGradients);

    public bool isOutputLayer() => isOutputLayer;
}


public class Matrix{
    private float[,] matrix;
    private int rowNum, colNum;
    public Matrix(int rowNum, int colNum) {
        this.matrix = new float[rowNum, colNum];
        this.rowNum = rowNum;
        this.colNum = colNum;
    }

    public Matrix(float[] vector) {
        this.rowNum = 1;
        this.colNum = vector.Length;
        this.matrix = new float[rowNum, colNum];

        for (int i = 0; i < colNum; i++) 
            matrix[0, i] = vector[i];

    }

    public Matrix(float[,] matrix) => this.matrix = matrix;


    public float this[int row, int col] {
        get => matrix[row, col];
        set => matrix[row, col] = value;
    }

    public float[] this[int row] {
        get {
            float[] result = new float[colNum];
            for (int j = 0; j < colNum; j++)
                result[j] = matrix[row, j];
            return result;
        }
        set {
            for (int j = 0; j < colNum; j++)
                matrix[row, j] = value[j];
        }
    }

    public static Matrix operator *(Matrix a, Matrix b) {

        int aRowNum = a.GetLength(0); 
        int bRowNum = b.GetLength(0); 
        int aColNum = a.GetLength(1);
        int bColNum = b.GetLength(1);

        if (aColNum != bRowNum)
            throw new ArgumentException("matrix dimensions must match for multiplication");

        Matrix result = new Matrix(aRowNum, bColNum);

        for (int i = 0; i < aRowNum; i++) {
            for (int j = 0; j < bColNum; j++) {
                
                float sum = 0f;

                for (int k = 0; k < aColNum; k++) 
                    sum += a[i, k] * b[k, j];
                    
                result[i, j] = sum;
            }
        }
        return result;
    }

    public static Matrix operator +(Matrix a, Matrix b){
        int aRowNum = a.GetLength(0); 
        int bRowNum = b.GetLength(0); 
        int aColNum = a.GetLength(1);
        int bColNum = b.GetLength(1);

        if (aRowNum != bRowNum || aColNum != bColNum)
            throw new ArgumentException("matrix dimensions must match for additio");

        Matrix result = new Matrix(aRowNum, aColNum);

        for (int i = 0; i < aRowNum; i++){
            for (int j = 0; j < aColNum; j++){
                result[i, j] = a[i, j] + b[i, j];
            }
        }

        return result;
    }

    public Matrix Apply(Func<float, float> func) {
        Matrix result = new Matrix(colNum, rowNum);

        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                result[i, j] = func(matrix[i, j]);
            }
        }
        return result;
    }

    public Matrix HeInit() {
        for (int i = 0; i < rowNum; i++) {
            float[] row = MathsUtils.HeInit(colNum);
            for (int j = 0; j < colNum; j++){
                matrix[i, j] = row[j];
            }
        }
        return this;
    }

    public Matrix Transpose() {
        Matrix result = new Matrix(colNum, rowNum);
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                result[j, i] = matrix[i, j];
            }
        }
        return result;
    }

    public Matrix Hadamard(Matrix matrixB) {
        int bRowNum = matrixB.GetLength(0); 
        int bColNum = matrixB.GetLength(1); 

        if (this.rowNum != bRowNum || this.colNum != bColNum)
            throw new ArgumentException("matrix dimensions must match for additio");

        Matrix result = new Matrix(this.rowNum, this.colNum);
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                result[i, j] *= matrixB[i, j];
            }
        }
        return result;
    }

    public int GetLength(int dimension)
    {
        if (dimension == 0) return rowNum;
        if (dimension == 1) return colNum;
        throw new ArgumentException("invalid dimension, use 0 for rows, 1 for columns");
    }
    

}


public static class MathsUtils {
    private static Random random = new Random();

    public static double NextGaussian(double mean = 0, double stddev = 1){   
        double epsilon = 1e-10;
        double r1 = random.NextDouble() + epsilon; 
        double r2 = random.NextDouble() + epsilon;
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(r1)) *  Math.Sin(2.0 * Math.PI * r2); 
        return (float)(mean + stddev * randStdNormal);
    }

    public static float[] HeInit(int numOfInputs){
        float[] weights = new float[numOfInputs];
        float stdDev = Math.Sqrt(2.0 / numOfInputs);
        for(int i = 0; i < numOfInputs; i++)
            weights[i] = NextGaussian(0, stdDev);
        
        return weights;
    }

    public static Matrix Softmax(Matrix input) {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            float[] row = input[i];

            float max = row.Max();

            float[] exps = new float[cols];
            float sum = 0f;

            for (int j = 0; j < cols; j++) {
                exps[j] = (float)Math.Exp(row[j] - max);
                sum += exps[j];
            }

            for (int j = 0; j < cols; j++) {
                result[i, j] = exps[j] / sum;
            }
        }

        return result;
    }

    private static float LeakyReLUnegativeGradient = 0.001f;

    public float LeakyReLU(float value) => 
        value > 0 ? value : LeakyReLUnegativeGradient * value;

    public float LeakyReLUDeriv(float value) =>
        value > 0 ? 1 : LeakyReLUnegativeGradient;
}


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