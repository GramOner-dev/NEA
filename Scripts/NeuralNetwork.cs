using System;

public class Network
{
    private Layer[] layers;
    private Matrix inputGradients;
    public Network(int[] topology)
    {
        layers = new Layer[topology.Length - 1];
        inputGradients = new Matrix(topology[0]);

        for (int i = 1; i < topology.Length; i++)
        {
            bool isOutputLayer = i == topology.Length - 1;
            layers[i - 1] = new Layer(topology[i - 1], topology[i], isOutputLayer);
        }
    }

    public Matrix Forward(Matrix input)
    {
        Matrix currentOutput = input;

        for (int i = 0; i < layers.Length; i++)
            currentOutput = layers[i].Forward(currentOutput);

        return currentOutput;
    }

    private Matrix RecursiveBackprop(int layerIndex, Matrix dLdOutput)
    {
        if (layerIndex < 0)
            return dLdOutput;

        Matrix dLdInput = layers[layerIndex].Backward(dLdOutput);
        layers[layerIndex].UpdateParams();
        return RecursiveBackprop(layerIndex - 1, dLdInput);
    }

    public Matrix Backward(Matrix dLdOutput)
    {
        inputGradients = RecursiveBackprop(layers.Length - 1, dLdOutput);
        return inputGradients;
    }
}

public class Layer
{

    private Matrix input, logits, output;
    private bool isOutputLayer;
    private LayerNormalizer LayerNorm;
    private WeightBiasPair Weights;
    public Layer(int numOfInputs, int numOfNeurons, bool isOutputLayer)
    {
        Weights = new WeightBiasPair(numOfInputs, numOfNeurons, InitType.He);

        input = new Matrix(numOfInputs);
        logits = new Matrix(numOfInputs, numOfNeurons);
        output = new Matrix(numOfNeurons);

        this.isOutputLayer = isOutputLayer;

        LayerNorm = new LayerNormalizer(numOfNeurons);
    }

    public Matrix Forward(Matrix input)
    {
        this.input = input;
        logits = Weights.Forward(input);
        if (!isOutputLayer)
        {
            Matrix normalized = LayerNorm.Forward(logits);
            output = normalized.Apply(MathsUtils.LeakyReLU);
        }
        else
        {
            output = logits;
        }
        return output;
    }

    public Matrix Backward(Matrix nextLayerGradients)
    {
        Matrix dLdPreActivation;

        if (!isOutputLayer)
        {
            Matrix normalized = LayerNorm.GetNormalizedInputs();
            Matrix dLdActivation = nextLayerGradients.Hadamard(normalized.Apply(MathsUtils.LeakyReLUDeriv));

            dLdPreActivation = LayerNorm.Backward(dLdActivation);
        }
        else
        {
            dLdPreActivation = nextLayerGradients;
        }
        Matrix dLdWeights = dLdPreActivation * input.Transpose();
        Weights.SetWeightGradients(dLdWeights);
        Weights.SetBiasGradients(dLdPreActivation);

        Matrix dLdInput = Weights.GetWeights().Transpose() * dLdPreActivation;
        return dLdInput;
    }

    public void UpdateParams()
    {
        Weights.Update();
    }
}
