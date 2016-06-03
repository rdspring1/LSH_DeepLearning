package dl.nn;

public class LogisticNeuronLayer extends HiddenLayer
{
    public LogisticNeuronLayer(int prev_layer_size, int layer_size, double L2)
    {
        super(prev_layer_size, layer_size, L2);
    }

    public HiddenLayer clone()
    {
        LogisticNeuronLayer copy = new LogisticNeuronLayer(m_prev_layer_size, m_layer_size, L2_Lambda);
        copy.m_theta = this.m_theta;
        copy.m_pos = this.m_pos;
        return copy;
    }

    // Random Weight Initialization
    protected double weightInitialization()
    {
        double interval = 4.0*Math.sqrt(6.0 / (m_prev_layer_size + m_layer_size));
        return Util.rand.nextDouble() * (2*interval) - interval;
    }

    // Activation Function
    protected double[] activationFunction(double[] input)
    {
        double[] output = new double[input.length];
        for(int idx = 0; idx < output.length; ++idx)
        {
           output[idx] = 1.0 / (1.0 + Math.exp(-input[idx]));
        }
        return output;
    }

    // Derivative Function
    protected double derivative(double input)
    {
        double negative_exp = Math.exp(-input);
        return negative_exp / Math.pow((1 + negative_exp), 2.0);
    }
}
