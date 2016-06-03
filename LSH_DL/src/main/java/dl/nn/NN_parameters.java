package dl.nn;


import dl.lsh.CosineDistance;
import dl.lsh.HashBuckets;
import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Created by sml1 on 9/23/15.
 */
public class NN_parameters
{
    // Neural Network Structure
    private int m_epoch_offset;
    private List<NeuronLayer> m_layers;

    private int[] m_weight_idx;
    private int[] m_bias_idx;
    private int m_size = 0;
    private int m_layer_count = 0;
    private int[] m_layer_row;
    private int[] m_layer_col;

    // Stochastic Gradient Descent
    private double[] m_theta;
    private double[] m_gradient;

    // Momentum
    private double[] m_momentum;
    private double m_momentum_lambda = 0.50;
    private final double momentum_max = 0.90;
    private final double momentum_rate = 1.00;

    // Learning Rate - Adagrad
    private final double m_learning_rate;
    private double[] m_learning_rates;

    // LSH
    private List<HashBuckets> m_tables;
    private final int[] m_poolDim;
    private final int[] m_b;
    private final int[] m_L;
    private double[] m_size_limit;

    // Create a NN_parameters object for a given neural network
    public NN_parameters(List<NeuronLayer> NN_structure, final int[] poolDim, final int[] b, final int[] L, final double learning_rate, double[] size_limit)
    {
        m_layers = NN_structure;
        m_learning_rate = learning_rate;
        m_poolDim = poolDim;
        m_b = b;
        m_L = L;
        m_size_limit = size_limit;
        m_tables = new ArrayList<>();
        construct(NN_structure);
        weight_initialization(NN_structure);
        createLSHTable(m_tables, poolDim, b, L, size_limit);
        System.out.println("Finished Initializing Parameters");
    }

    // Load parameters from saved file
    public NN_parameters(BufferedReader reader, List<NeuronLayer> NN_structure, final int[] poolDim, final int[] b, final int[] L, final double learning_rate, double[] size_limit) throws IOException
    {
        m_layers = NN_structure;
        m_learning_rate = learning_rate;
        m_poolDim = poolDim;
        m_b = b;
        m_L = L;
        m_size_limit = size_limit;
        m_tables = new ArrayList<>();
        construct(NN_structure);

        // Load model and parameters
        m_epoch_offset = Integer.parseInt(reader.readLine());
        load_model(NN_structure, reader, m_theta);
        load_model(NN_structure, reader, m_momentum);
        load_model(NN_structure, reader, m_learning_rates);

        createLSHTable(m_tables, poolDim, b, L, size_limit);
        System.out.println("Finished Initializing Parameters");
    }

    /*
        Create an empty duplicate of the other NN_parameters object
     */
    public double[] copy()
    {
        return Arrays.copyOf(m_theta, m_theta.length);
    }

    public void copy(double[] theta)
    {
        assert(theta.length == m_theta.length);
        m_theta = Arrays.copyOf(theta, theta.length);
        Arrays.fill(m_momentum, 0.0);
        Arrays.fill(m_learning_rates, 0.0);
    }

    public void construct(List<NeuronLayer> NN_structure)
    {
        m_weight_idx = new int[NN_structure.size()];
        m_bias_idx = new int[NN_structure.size()];
        m_layer_row = new int[NN_structure.size()];
        m_layer_col = new int[NN_structure.size()];

        for(NeuronLayer l : NN_structure)
        {
            m_layer_row[m_layer_count] = l.m_layer_size;
            m_layer_col[m_layer_count] = l.m_prev_layer_size;

            m_weight_idx[m_layer_count] = m_size;
            m_size += l.numWeights();
            m_bias_idx[m_layer_count] = m_size;
            m_size += l.numBias();
            l.m_pos = m_layer_count++;
            l.m_theta = this;
        }
        m_theta = new double[m_size];
        m_gradient = new double[m_size];
        m_momentum = new double[m_size];
        m_learning_rates = new double[m_size];
    }

    public int epoch_offset()
    {
        return m_epoch_offset;
    }

    public void save_model(int epoch, BufferedWriter writer) throws IOException
    {
        writer.write(Long.toString(epoch + m_epoch_offset));
        writer.newLine();
        save_model(writer, m_theta);
        save_model(writer, m_momentum);
        save_model(writer, m_learning_rates);
        writer.close();
    }

    private void save_model(BufferedWriter writer, double[] array) throws IOException
    {
        DecimalFormat df = new DecimalFormat("#.##########");
        final String SPACE =  " ";

        int global_idx = -1;
        for(NeuronLayer l : m_layers)
        {
            for(int idx = 0; idx < l.m_layer_size; ++idx)
            {
                for(int jdx = 0; jdx < l.m_prev_layer_size; ++jdx)
                {
                    writer.write(df.format(array[++global_idx]));
                    writer.write(SPACE);
                }
                writer.newLine();
            }

            for(int idx = 0; idx < l.m_layer_size; ++idx)
            {
                writer.write(df.format(array[++global_idx]));
                writer.write(SPACE);
            }
            writer.newLine();
        }
        assert(global_idx == m_size-1);
    }

    private void load_model(List<NeuronLayer> NN_structure, BufferedReader reader, double[] array) throws IOException
    {
        int global_idx = -1;
        for(NeuronLayer l : NN_structure)
        {
            for(int idx = 0; idx < l.m_layer_size; ++idx)
            {
                String[] node = reader.readLine().trim().split("\\s+");
                assert(node.length == l.m_prev_layer_size);
                for(String weight : node)
                {
                    array[++global_idx] = Double.parseDouble(weight);
                }
            }

            String[] biases = reader.readLine().trim().split("\\s+");
            assert(biases.length == l.m_layer_size);
            for(String bias : biases)
            {
                array[++global_idx] = Double.parseDouble(bias);
            }
        }
        assert(global_idx == m_size-1);
    }

    private void weight_initialization(List<NeuronLayer> NN_structure)
    {
        int global_idx = -1;
        for(NeuronLayer l : NN_structure)
        {
            for(int idx = 0; idx < l.m_layer_size; ++idx)
            {
                for(int jdx = 0; jdx < l.m_prev_layer_size; ++jdx)
                {
                    m_theta[++global_idx] = l.weightInitialization();
                }
            }
            global_idx += l.m_layer_size;
        }
        assert(global_idx == m_size-1);
    }

    public List<int[]> computeHashes(List<DoubleMatrix> data)
    {
        final int interval = data.size() / 10;
        List<int[]> hashes = new ArrayList<>();
        for(int idx = 0; idx < data.size(); ++idx)
        {
            if(idx % interval == 0)
            {
                System.out.println("Completed " + idx + " / " + data.size());
            }

            hashes.add(m_tables.get(0).generateHashSignature(data.get(idx)));
        }
        return hashes;
    }

    public DoubleMatrix getWeight(int layer, int node)
    {
         assert(layer >= 0 && layer < m_layer_count);
         assert(node >= 0 && node < m_layer_row[layer]);

         return Util.vectorize(m_theta, m_weight_idx[layer] + node * m_layer_col[layer], m_layer_col[layer]);
    }

    public void rebuildTables()
    {
        int global_idx = 0;
        for(int layer_idx = 0; layer_idx < m_layer_count-1; ++layer_idx)
        {
            m_tables.get(layer_idx).clear();
            for(int idx = 0; idx < m_layer_row[layer_idx] ; ++idx)
            {
                m_tables.get(layer_idx).LSHAdd(idx, Util.vectorize(m_theta, global_idx, m_layer_col[layer_idx]));
                global_idx += m_layer_col[layer_idx];
            }
            global_idx += m_layer_row[layer_idx];
        }
    }

    public void createLSHTable(List<HashBuckets> tables, int[] poolDim, int[] b, int[] L, final double[] size_limit)
    {
        int global_idx = 0;
        for(int layer_idx = 0; layer_idx < m_layer_count-1; ++layer_idx)
        {
            HashBuckets table = new HashBuckets(size_limit[layer_idx] * m_layer_row[layer_idx], poolDim[layer_idx], L[layer_idx], new CosineDistance(b[layer_idx], L[layer_idx], m_layer_col[layer_idx] / poolDim[layer_idx]));
            for(int idx = 0; idx < m_layer_row[layer_idx] ; ++idx)
            {
                table.LSHAdd(idx, Util.vectorize(m_theta, global_idx, m_layer_col[layer_idx]));
                global_idx += m_layer_col[layer_idx];
            }
            tables.add(table);
            global_idx += m_layer_row[layer_idx];
        }
    }

    public Set<Integer> retrieveNodes(int layer, DoubleMatrix input)
    {
        //return m_tables.get(layer).LSHUnion(input);
        return m_tables.get(layer).histogramLSH(input);
    }

    public Set<Integer> retrieveNodes(int layer, int[] hashes)
    {
        //return m_tables.get(layer).LSHUnion(hashes);
        return m_tables.get(layer).histogramLSH(hashes);
    }

    public void timeStep()
    {
        m_momentum_lambda *= momentum_rate;
        m_momentum_lambda = Math.min(m_momentum_lambda, momentum_max);
    }

    public int size()
    {
        return m_size;
    }

    public double getGradient(int idx)
    {
        assert(idx >= 0 && idx < m_theta.length);
        return m_momentum[idx] / m_learning_rate;
    }

    public double getTheta(int idx)
    {
        assert(idx >= 0 && idx < m_theta.length);
        return m_theta[idx];
    }

    public void setTheta(int idx, double value)
    {
        assert(idx >= 0 && idx < m_theta.length);
        m_theta[idx] = value;
    }

    public double getWeight(int layer, int row, int col)
    {
        assert(layer >= 0 && layer < m_layer_count);
        assert(row >= 0 && row < m_layer_row[layer]);
        assert(col >= 0 && col < m_layer_col[layer]);

        int idx = row * m_layer_col[layer] + col;
        return m_theta[m_weight_idx[layer] + idx];
    }

    public DoubleMatrix getWeightVector(int layer, int node_idx)
    {
        assert(layer >= 0 && layer < m_layer_count);
        assert(node_idx >= 0 && node_idx < m_layer_row[layer]);
        return Util.vectorize(m_theta, m_weight_idx[layer] + node_idx * m_layer_col[layer], m_layer_col[layer]);
    }

    public void setWeight(int layer, int row, int col, double value)
    {
        assert(layer >= 0 && layer < m_layer_count);
        assert(row >= 0 && row < m_layer_row[layer]);
        assert(col >= 0 && col < m_layer_col[layer]);

        int idx = row * m_layer_col[layer] + col;
        m_theta[m_weight_idx[layer] + idx] = value;
    }

    public double getBias(int layer, int idx)
    {
        assert(layer >= 0 && layer < m_layer_count);
        assert(idx >= 0 && idx < m_layer_row[layer]);

        return m_theta[m_bias_idx[layer] + idx];
    }

    public void setBias(int layer, int idx, double value)
    {
        assert(layer >= 0 && layer < m_layer_count);
        assert(idx >= 0 && idx < m_layer_row[layer]);

        m_theta[m_bias_idx[layer] + idx] = value;
    }

    public double L2_regularization()
    {
        double L2 = 0.0;
        for (int layer_idx = 0; layer_idx < m_layer_count; ++layer_idx)
        {
            for(int idx = m_weight_idx[layer_idx]; idx < m_bias_idx[layer_idx]; ++idx)
            {
                L2 += Math.pow(m_theta[idx], 2.0);
            }
        }
        return 0.5 * L2;
    }

    public int weightOffset(int layer, int row, int column)
    {
        assert(layer >= 0 && layer < m_layer_count);
        assert(row >= 0 && row < m_layer_row[layer]);
        assert(column >= 0 && column < m_layer_col[layer]);

        int idx = row * m_layer_col[layer] + column;
        return m_weight_idx[layer] + idx;
    }

    public int biasOffset(int layer, int idx)
    {
        assert(layer >= 0 && layer < m_layer_count);
        assert(idx >= 0 && idx < m_layer_row[layer]);

        return m_bias_idx[layer] + idx;
    }

    public void stochasticGradientDescent(int idx, double gradient)
    {
        m_gradient[idx] = gradient;
        m_learning_rates[idx] += Math.pow(gradient, 2.0);
        double learning_rate = m_learning_rate / (1e-6 + Math.sqrt(m_learning_rates[idx]));
        m_momentum[idx] *= m_momentum_lambda;
        m_momentum[idx] += learning_rate * gradient;
        m_theta[idx] -= m_momentum[idx];
    }

    public void clear_gradient()
    {
        Arrays.fill(m_gradient, 0);
    }

    public void print_active_nodes(String dataset, String filename, double threshold) throws Exception
    {
        final int linesize = 25;

        BufferedWriter writer = new BufferedWriter(new FileWriter(Util.DATAPATH + dataset + "/" + dataset + "_" + filename, true));
        StringBuilder string = new StringBuilder();
        string.append("[");
        int count = 0;
        for(int layer = 0; layer < m_layer_row.length; ++layer)
        {
            int layer_size = m_layer_row[layer];
            int[] grad_count = new int[layer_size];
            for(int idx = 0; idx < layer_size; ++idx)
            {
                int pos = idx * m_layer_col[layer];
                for(int jdx = 0; jdx < m_layer_col[layer]; ++jdx)
                {
                    if(Math.abs(m_gradient[pos+jdx]) > 0)
                    {
                        ++grad_count[idx];
                    }
                }
            }

            for(int idx = 0; idx < layer_size; ++idx)
            {
                int value = (grad_count[idx] >= threshold * layer_size)? 1 : 0;
                string.append(value);
                if(!(layer == m_layer_row.length-1 && idx == layer_size-1))
                {
                    if(count <= linesize)
                    {
                        string.append(", ");
                    }
                    else
                    {
                        string.append("\n");
                        count = 0;
                    }
                }
                ++count;
            }
        }
        string.append("]\n");
        writer.write(string.toString());
        writer.flush();
        writer.close();
    }
}
