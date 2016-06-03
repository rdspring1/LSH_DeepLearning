package dl.nn;

import org.jblas.DoubleMatrix;
import java.util.*;

public class NeuralNetwork
{
    private List<NeuronLayer> m_layers;
    private LinkedList<HiddenLayer> m_hidden_layers;
    private double L2_lambda;
    private ICostFunction m_cf;
    private NN_parameters m_params;
    private double m_cost;
    protected double m_train_correct;

    public NeuralNetwork(NN_parameters params, List<NeuronLayer> layers, LinkedList<HiddenLayer> hiddenLayers, double L2, ICostFunction cf)
    {
        L2_lambda = L2;
        m_cf = cf;
        m_params = params;
        m_hidden_layers = hiddenLayers;
        m_layers = layers;
    }

    public long calculateActiveNodes()
    {
        long total = 0;
        for(HiddenLayer l : m_hidden_layers)
        {
            total += l.m_total_nn_set_size;
            l.m_total_nn_set_size = 0;
        }
        total += m_layers.get(m_layers.size()-1).m_layer_size;
        return total;
    }

    public long calculateMultiplications()
    {
        long total = 0;
        for(HiddenLayer l : m_hidden_layers)
        {
            total += l.m_total_multiplication;
            l.m_total_multiplication = 0;
        }
        total += m_layers.get(m_layers.size()-1).numWeights();
        return total;
    }

    public double test(List<int[]> input_hashes, List<DoubleMatrix> data, double[] labels)
    {
        double[][] y_hat = new double[labels.length][];
        for(int idx = 0; idx < labels.length; ++idx)
        {
            y_hat[idx] = forwardPropagation(data.get(idx), input_hashes.get(idx), false);
        }
        return m_cf.accuracy(y_hat, labels);
    }

    public double getGradient(int idx)
    {
        return m_params.getGradient(idx);
    }

    public double[] copyTheta()
    {
        return m_params.copy();
    }

    public double getCost()
    {
        return m_cost;
    }

    public int numTheta()
    {
        return m_params.size();
    }

    public double getTheta(int idx)
    {
        return m_params.getTheta(idx);
    }

    public void setTheta(double[] params)
    {
        m_params.copy(params);
    }

    public void setTheta(int idx, double value)
    {
        m_params.setTheta(idx, value);
    }

    public List<int[]> computeHashes(List<DoubleMatrix> data)
    {
        return m_params.computeHashes(data);
    }

    public void updateHashTables(int miniBatch_size)
    {
        m_hidden_layers.forEach(e -> e.updateHashTables(miniBatch_size));
    }

    public void execute(int[] hashes, DoubleMatrix input, double labels, boolean training)
    {
        double[] y_hat = forwardPropagation(input, hashes, training);
        backPropagation(y_hat, labels); // Calculate Cost and Gradient
        m_train_correct += m_cf.correct(y_hat, labels);
    }

    private void backPropagation(double[] y_hat, double labels)
    {
        // square loss function
        m_cost = m_cf.costFunction(y_hat, labels) + L2_lambda * m_params.L2_regularization();

        NeuronLayer outputLayer = m_layers.get(m_layers.size()-1);

        // cost function derivatives
        double[] delta = m_cf.outputDelta(y_hat, labels, outputLayer);

        // Calculate the gradient for the output layer
        ListIterator<NeuronLayer> it = m_layers.listIterator(m_layers.size());
        // Calculate the delta for the hidden layers
        while (it.hasPrevious())
        {
            delta = it.previous().calculateDelta(delta);
        }

        // Calculate the gradient for the output layer
        it = m_layers.listIterator(m_layers.size());
        while (it.hasPrevious())
        {
            it.previous().calculateGradient();
        }
    }

    private double[] forwardPropagation(DoubleMatrix input, int[] hashes, boolean training)
    {
        Iterator<HiddenLayer> iterator = m_hidden_layers.iterator();
        double[] data = iterator.next().forwardPropagation(input, hashes, training);
        while (iterator.hasNext())
        {
            data = iterator.next().forwardPropagation(data, training);
        }
        return m_layers.get(m_layers.size() - 1).forwardPropagation(data);
    }
}
