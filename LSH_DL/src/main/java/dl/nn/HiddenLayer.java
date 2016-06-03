package dl.nn;

import org.jblas.DoubleMatrix;

import java.util.*;

public abstract class HiddenLayer extends NeuronLayer
{
    protected Set<Integer> m_node_set;
    protected Set<Integer> m_total_node_set;
    protected long m_total_nn_set_size;
    protected long m_total_multiplication;

    public HiddenLayer(int prev_layer_size, int layer_size, double L2)
    {
        super(prev_layer_size, layer_size, L2);
        m_total_node_set = new HashSet<>();
        m_delta = new double[m_layer_size];
    }

    public abstract HiddenLayer clone();

    // Derivative Function
    protected abstract double derivative(double input);

    public double[] forwardPropagation(DoubleMatrix input, Set<Integer> nn_node_set, boolean training)
    {
        assert(nn_node_set.size() <= m_layer_size);
        assert(input.length == m_prev_layer_size);

        m_input = input.toArray();
        m_node_set = nn_node_set;

        if(training)
        {
            m_total_nn_set_size += m_node_set.size();
            m_total_multiplication += m_node_set.size() * m_prev_layer_size;
        }

        Arrays.fill(m_weightedSum, 0.0);
        for(int idx : nn_node_set)
        {
            m_weightedSum[idx] = m_theta.getWeightVector(m_pos, idx).dot(input) + m_theta.getBias(m_pos, idx);
        }
        return activationFunction();
    }

    public double[] forwardPropagation(double[] input)
    {
        return forwardPropagation(Util.vectorize(input), false);
    }

    public double[] forwardPropagation(double[] input, boolean training)
    {
        return forwardPropagation(Util.vectorize(input), training);
    }

    public double[] forwardPropagation(DoubleMatrix input, boolean training)
    {
        return forwardPropagation(input, m_theta.retrieveNodes(m_pos, input), training);
    }

    public double[] forwardPropagation(DoubleMatrix input, int[] hashes, boolean training)
    {
        return forwardPropagation(input, m_theta.retrieveNodes(m_pos, hashes), training);
    }

    public double[] calculateDelta(final double[] prev_layer_delta)
    {
        Arrays.fill(m_delta, 0.0);
        for(int idx : m_node_set)
        {
            for(int jdx = 0; jdx < prev_layer_delta.length; ++jdx)
            {
                m_delta[idx] += m_theta.getWeight(m_pos+1, jdx, idx) * prev_layer_delta[jdx];
            }
            m_delta[idx] *= derivative(m_weightedSum[idx]);
        }
        return m_delta;
    }

    public void calculateGradient()
    {
        assert(m_delta.length == m_layer_size);

        for(int idx : m_node_set)
        {
            // Set Weight Gradient
            for(int jdx = 0; jdx < m_prev_layer_size; ++jdx)
            {
                m_theta.stochasticGradientDescent(m_theta.weightOffset(m_pos, idx, jdx), m_delta[idx] * m_input[jdx]);
            }

            // Set Bias Gradient
            m_theta.stochasticGradientDescent(m_theta.biasOffset(m_pos, idx), m_delta[idx]);
        }
    }

    public void updateHashTables(double size)
    {
        System.out.println(m_pos + " : " + m_total_nn_set_size / size);
    }
}
