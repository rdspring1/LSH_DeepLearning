package dl.nn;

public class ReLUNeuronLayer extends HiddenLayer
{
    public ReLUNeuronLayer(int prev_layer_size, int layer_size, double L2)
    {
        super(prev_layer_size, layer_size, L2);
    }

    public HiddenLayer clone()
    {
        ReLUNeuronLayer copy = new ReLUNeuronLayer(m_prev_layer_size, m_layer_size, L2_Lambda);
        copy.m_theta = this.m_theta;
        copy.m_pos = this.m_pos;
        return copy;
    }

    // Random Weight Initialization
    protected double weightInitialization()
    {
        double interval = 2.0*Math.sqrt(6.0 / (m_prev_layer_size + m_layer_size));
        return Util.rand.nextDouble() * (2*interval) - interval;
    }

    // Activation Function
    protected double[] activationFunction(double[] input)
    {
        double[] output = new double[input.length];
        for(int idx = 0; idx < output.length; ++idx)
        {
            output[idx] = Math.max(input[idx], 0.0);
        }
        return output;
    }

    // Derivative Function
    protected double derivative(double input)
    {
        return (input > 0) ? 1.0 : 0.0;
    }
}
