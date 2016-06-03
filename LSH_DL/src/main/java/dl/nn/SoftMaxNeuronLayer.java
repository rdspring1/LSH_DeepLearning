package dl.nn;

public class SoftMaxNeuronLayer extends NeuronLayer
{
    public SoftMaxNeuronLayer(int prev_layer_size, int layer_size, double L2)
    {
        super(prev_layer_size, layer_size, L2);
    }

    public NeuronLayer clone()
    {
        SoftMaxNeuronLayer copy = new SoftMaxNeuronLayer(m_prev_layer_size, m_layer_size, L2_Lambda);
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
        double sum = 0.0;
        double[] output = new double[input.length];
        for(int idx = 0; idx < input.length; ++idx)
        {
            output[idx] = Math.exp(input[idx]);
            sum += output[idx];
        }

        for(int idx = 0; idx < output.length; ++idx)
        {
            output[idx] /= sum;
        }
        return output;
    }

    public double[] forwardPropagation(double[] input)
    {
        assert(input.length == m_prev_layer_size);
        m_input = input;

        for(int jdx = 0; jdx < m_layer_size; ++jdx)
        {
            m_weightedSum[jdx] = 0.0;
            for(int idx = 0; idx < m_prev_layer_size; ++idx)
            {
                m_weightedSum[jdx] += m_theta.getWeight(m_pos, jdx, idx)  * m_input[idx];
            }
            m_weightedSum[jdx] += m_theta.getBias(m_pos, jdx);
        }
        return activationFunction();
    }

    public double[] calculateDelta(double[] prev_layer_delta)
    {
        assert(prev_layer_delta.length == m_layer_size);
        m_delta = prev_layer_delta;
        return m_delta;
    }

    public void calculateGradient()
    {
        assert(m_delta.length == m_layer_size);

        for(int idx = 0; idx < m_layer_size; ++idx)
        {
            // Set Weight Gradient
            for(int jdx = 0; jdx < m_prev_layer_size; ++jdx)
            {
                m_theta.stochasticGradientDescent(m_theta.weightOffset(m_pos, idx, jdx), m_delta[idx] * m_input[jdx] + L2_Lambda * m_theta.getWeight(m_pos, idx, jdx));
            }

            // Set Bias Gradient
            m_theta.stochasticGradientDescent(m_theta.biasOffset(m_pos, idx), m_delta[idx]);
        }
    }
}