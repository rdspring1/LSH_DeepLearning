package dl.nn;

// Input: A(l) Params: W(l) b(l) Output: Z(l+1) A(l+1)
public abstract class NeuronLayer implements Cloneable
{
    public int m_pos = -1;
    public NN_parameters m_theta;

    protected double[] m_input;
    protected double[] m_weightedSum;
    protected double[] m_delta;

    protected int m_prev_layer_size;
    protected int m_layer_size;
    protected double L2_Lambda;

    public NeuronLayer(int prev_layer_size, int layer_size, double L2)
    {
        m_prev_layer_size = prev_layer_size;
        m_layer_size = layer_size;
        L2_Lambda = L2;
        m_weightedSum = new double[m_layer_size];
    }

    public abstract NeuronLayer clone();

    // Random Weight Initialization
    protected abstract double weightInitialization();

    // Activation Function
    protected abstract double[] activationFunction(double[] input);

    public double[] activationFunction()
    {
        return activationFunction(m_weightedSum);
    }

    public int numWeights()
    {
        return m_prev_layer_size * m_layer_size;
    }

    public int numBias()
    {
        return m_layer_size;
    }

    public abstract double[] forwardPropagation(double[] input);

    public abstract double[] calculateDelta(double[] prev_layer_delta);

    public abstract void calculateGradient();
}
