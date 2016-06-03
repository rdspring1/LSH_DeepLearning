package dl.nn;

import org.jblas.DoubleMatrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class NeuralCoordinator
{
    private List<NeuralNetwork> m_networks;
    private NN_parameters m_params;
    private String m_modelTitle;
    private String m_model_path;
    private String m_train_path;
    private String m_test_path;
    private double m_total_nodes;
    private final int update_threshold = 20;

    public NeuralCoordinator(String model_title, String title, String dataset, NN_parameters params, List<NeuronLayer> layers, LinkedList<HiddenLayer> hiddenLayers, double L2, ICostFunction cf) throws IOException
    {
        m_modelTitle = model_title;
        m_model_path = Util.DATAPATH + dataset + "/" + Util.MODEL + title + "_" + model_title;
        m_train_path = Util.DATAPATH + dataset + "/" + Util.TRAIN + title;
        m_test_path = Util.DATAPATH + dataset + "/" + Util.TEST + title;

        for(NeuronLayer layer : layers)
        {
            m_total_nodes += layer.m_layer_size;
        }

        m_params = params;
        m_networks = new ArrayList<>(Util.LAYER_THREADS);
        m_networks.add(new NeuralNetwork(params, layers, hiddenLayers, L2, cf));
        for(int idx = 1; idx < Util.LAYER_THREADS; ++idx)
        {
            LinkedList<HiddenLayer> hiddenLayers1 = new LinkedList<>();
            hiddenLayers.forEach(e -> hiddenLayers1.add(e.clone()));

            List<NeuronLayer> layers1 = new LinkedList<>();
            layers1.addAll(hiddenLayers1);
            layers1.add(layers.get(layers.size()-1).clone());
            m_networks.add(new NeuralNetwork(params, layers1, hiddenLayers1, L2, cf));
        }
    }

    private List<Integer> initIndices(int length)
    {
        List<Integer> indices = new ArrayList<>();
        for(int idx = 0; idx < length; ++idx)
        {
            indices.add(idx);
        }
        return indices;
    }

    private void shuffle(List<Integer> indices)
    {
        for(int idx = 0; idx < indices.size(); ++idx)
        {
            int rand = Util.rand.nextInt(indices.size());
            int value = indices.get(idx);
            indices.set(idx, indices.get(rand));
            indices.set(rand, value);
        }
    }

    public void test(List<DoubleMatrix> data, double[] labels)
    {
        List<int[]> test_hashes = m_params.computeHashes(data);
        System.out.println("Finished Pre-Computing Training Hashes");
        System.out.println(m_networks.get(0).test(test_hashes, data, labels));
    }

    // training data, training labels
    public void train(final int max_epoch, List<DoubleMatrix> data, double[] labels, List<DoubleMatrix> test_data, double[] test_labels) throws Exception
    {
        assert(data.size() == labels.length);
        assert(test_data.size() == test_labels.length);

        List<int[]> input_hashes = m_params.computeHashes(data);
        System.out.println("Finished Pre-Computing Training Hashes");

        List<int[]> test_hashes = m_params.computeHashes(test_data);
        System.out.println("Finished Pre-Computing Testing Hashes");

        List<Integer> data_idx = initIndices(labels.length);
        final int m_examples_per_thread = data.size() / (Util.UPDATE_SIZE * Util.LAYER_THREADS);
        assert(data_idx.size() == labels.length);

        BufferedWriter train_writer = new BufferedWriter(new FileWriter(m_train_path, true));
        BufferedWriter test_writer = new BufferedWriter(new FileWriter(m_test_path, true));
        for(int epoch_count = 0; epoch_count < max_epoch; ++epoch_count)
        {
            m_params.clear_gradient();
            shuffle(data_idx);
            int count = 0;
            while(count < data_idx.size())
            {
                List<Thread> threads = new LinkedList<>();
                for(NeuralNetwork network : m_networks)
                {
                    if(count < data_idx.size())
                    {
                        int start = count;
                        count = Math.min(data_idx.size(), count + m_examples_per_thread);
                        int end = count;

                        Thread t = new Thread()
                        {
                            @Override
                            public void run()
                            {
                                for (int pos = start; pos < end; ++pos)
                                {
                                    network.execute(input_hashes.get(pos), data.get(pos), labels[pos], true);
                                }
                            }
                        };
                        t.start();
                        threads.add(t);
                    }
                }
                Util.join(threads);
                if(epoch_count <= update_threshold && epoch_count % (epoch_count / 10 + 1) == 0)
                {
                    m_params.rebuildTables();
                }

            }

            // Console Debug Output
            int epoch = m_params.epoch_offset() + epoch_count;
            //m_networks.stream().forEach(e -> e.updateHashTables(labels.length / Util.LAYER_THREADS));
            double activeNodes = calculateActiveNodes(m_total_nodes * data.size());
            double test_accuracy = m_networks.get(0).test(test_hashes, test_data, test_labels);
            System.out.println("Epoch " + epoch + " Accuracy: " + test_accuracy);

            // Test Output
            DecimalFormat df = new DecimalFormat("#.###");
            df.setRoundingMode(RoundingMode.FLOOR);
            test_writer.write(m_modelTitle + " " + epoch + " " + df.format(activeNodes) + " " + test_accuracy);
            test_writer.newLine();

            // Train Output
            train_writer.write(m_modelTitle + " " + epoch + " " + df.format(activeNodes) + " " + calculateTrainAccuracy(data.size()));
            train_writer.newLine();

            test_writer.flush();
            train_writer.flush();

            m_params.timeStep();
        }
        test_writer.close();
        train_writer.close();
        save_model(max_epoch, m_model_path);
    }

    public void save_model(int epoch, String path) throws IOException
    {
        m_params.save_model(epoch, Util.writerBZ2(path));
    }

    private double calculateTrainAccuracy(double size)
    {
        double count = 0;
        for(NeuralNetwork network : m_networks)
        {
            count += network.m_train_correct;
            network.m_train_correct = 0;
        }
        return count / size;
    }

    private double calculateActiveNodes(double total)
    {
        long active = 0;
        for(NeuralNetwork network : m_networks)
        {
            active += network.calculateActiveNodes();
        }
        return active / total;
    }

    private long calculateMultiplications()
    {
        long total = 0;
        for(NeuralNetwork network : m_networks)
        {
            total += network.calculateMultiplications();
        }
        return total;
    }
}
