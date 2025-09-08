using System;

public class Network
{
    private Layer[] layers;
    private Matrix inputGradients;
    public Network(int inputDim, int[] hidLayers, int outputDim)
    {
        int[] topology = CreateTopology(inputDim, hidLayers, outputDim);
        layers = new Layer[topology.Length - 1];
        InitLayers(topology);
        inputGradients = new Matrix(topology[0]);
    }

    private void InitLayers(int[] topology)
    {
        for (int i = 1; i < topology.Length; i++)
        {
            bool isOutputLayer = i == topology.Length - 1;
            layers[i - 1] = new Layer(topology[i - 1], topology[i], isOutputLayer);
        }
    }
    private int[] CreateTopology(int inputDim, int[] hidLayers, int outputDim)
    {
        List<int> list = new List<int>(hidLayers);
        list.Insert(0, inputDim);
        list.Add(outputDim);
        return list.ToArray();
    }

    public Matrix Forward(Matrix input)
    {
        Matrix currentOutput = input;

        for (int i = 0; i < layers.Length; i++)
        {
            currentOutput = layers[i].Forward(currentOutput);
        }
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
            Matrix normalized = LayerNorm.Forward(logits.Transpose());
            output = normalized.Apply(MathsUtils.LeakyReLU);
        }
        else
        {
            output = logits;
        }
        return output.Transpose();
    }

    public Matrix Backward(Matrix nextLayerGradients)
    {
        Matrix dLdPreActivation;

        if (!isOutputLayer)
        {
            Matrix normalized = LayerNorm.GetNormalizedInputs();
            Matrix dLdActivation = nextLayerGradients.Hadamard(normalized.Apply(MathsUtils.LeakyReLUDeriv).Transpose());
            dLdPreActivation = LayerNorm.Backward(dLdActivation);
        }
        else
        {
            dLdPreActivation = nextLayerGradients;
        }
        Matrix dLdWeights = dLdPreActivation.Transpose() * input.Transpose();

        Weights.SetWeightGradients(dLdWeights.Transpose());
        Weights.SetBiasGradients(dLdPreActivation.Transpose());

        Matrix dLdInput = Weights.GetWeights() * dLdPreActivation.Transpose();
        return dLdInput;
    }

    public void UpdateParams()
    {
        Weights.Update();
    }
}
