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
