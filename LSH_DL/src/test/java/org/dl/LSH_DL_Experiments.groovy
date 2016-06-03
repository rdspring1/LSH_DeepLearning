package org.dl

import dl.dataset.DLDataSet
import dl.dataset.MNISTDataSet
import dl.dataset.NORBDataSet

import dl.nn.CrossEntropy
import dl.nn.HiddenLayer
import dl.nn.NN_parameters
import dl.nn.NeuralCoordinator
import dl.nn.NeuronLayer
import dl.nn.ReLUNeuronLayer
import dl.nn.SoftMaxNeuronLayer
import dl.nn.Util
import org.apache.commons.lang3.tuple.Pair
import org.jblas.DoubleMatrix

class LSH_DL_Experiments extends GroovyTestCase
{
    // Grid Search
    private static final int min_layers = 1;
    private static final int max_layers = 1;
    private static final int hidden_layer_size = 1000;
    private static final int hidden_pool_size = hidden_layer_size * 0.1;

    // Neural Network Parameters
    private String title;
    private String dataset;
    private int training_size
    private int test_size
    private int inputLayer
    private int outputLayer
    private int k
    private int b = 6
    private int L = 100

    private final int max_epoch = 25
    private final double L2_Lambda = 0.003
    private int[] hiddenLayers
    private double[] learning_rates
    private final double[] size_limits = [0.05, 0.10, 0.25, 0.5, 0.75, 1.0]

    private LinkedList<HiddenLayer> hidden_layers
    private LinkedList<NeuronLayer> NN_layers

    String make_title()
    {
        StringBuilder titleBuilder = new StringBuilder()
        titleBuilder.append(dataset)
        titleBuilder.append('_')
        titleBuilder.append("LSH")
        titleBuilder.append('_')
        titleBuilder.append(inputLayer)
        titleBuilder.append('_')
        for(int idx = 0; idx < hiddenLayers.length; ++idx)
        {
            titleBuilder.append(hiddenLayers[idx])
            titleBuilder.append('_')
        }
        titleBuilder.append(outputLayer)
        title = titleBuilder.toString()
        return title;
    }

    void testMNIST()
    {
        dataset = "MNIST"
        training_size = 60000
        test_size = 10000
        inputLayer = 784
        outputLayer = 10
        k = 98
        learning_rates = [1e-2, 1e-2, 1e-2, 5e-3, 1e-3, 1e-3]

        // Read MNIST test and training data
        final String training_label_path = Util.DATAPATH + dataset + "/train-labels-idx1-ubyte"
        final String training_image_path = Util.DATAPATH + dataset + "/train-images-idx3-ubyte"
        final String test_label_path = Util.DATAPATH + dataset + "/t10k-labels-idx1-ubyte"
        final String test_image_path = Util.DATAPATH + dataset + "/t10k-images-idx3-ubyte"

        Pair<List<DoubleMatrix>, double[]> training = MNISTDataSet.loadDataSet(training_label_path, training_image_path)
        Pair<List<DoubleMatrix>, double[]> test = MNISTDataSet.loadDataSet(test_label_path, test_image_path)

        execute(training.getLeft(), training.getRight(), test.getLeft(), test.getRight());
    }

    void testNORB()
    {
        dataset = "NORB_SMALL"
        training_size = 20000
        test_size = 24300
        inputLayer = 2048
        outputLayer = 5
        k = 128
        learning_rates = [1e-2, 1e-2, 1e-2, 5e-3, 1e-3, 1e-3]

        // Read NORB training, validation, test data
        final String training_path = Util.DATAPATH + dataset + "/norb-small-train.bz2"
        final String test_path = Util.DATAPATH + dataset + "/norb-small-test.bz2"

        Pair<List<DoubleMatrix>, double[]> training = NORBDataSet.loadDataSet(Util.readerBZ2(training_path), training_size)
        Pair<List<DoubleMatrix>, double[]> test = NORBDataSet.loadDataSet(Util.readerBZ2(test_path), test_size)
        execute(training.getLeft(), training.getRight(), test.getLeft(), test.getRight());
    }

    void testRectangles()
    {
        test_size = 50000
        inputLayer = 784
        outputLayer = 2
        k = 98

        // Rectangles Data
        dataset = "Rectangles";
        training_size = 12000;
        final String training_path = Util.DATAPATH + dataset + "/rectangles_im_train.amat.bz2";
        final String test_path = Util.DATAPATH + dataset + "/rectangles_im_test.amat.bz2";
        learning_rates = [1e-2, 1e-2, 5e-3, 1e-3, 1e-3, 1e-3]

        Pair<List<DoubleMatrix>, double[]> training = DLDataSet.loadDataSet(Util.readerBZ2(training_path), training_size, inputLayer)
        Pair<List<DoubleMatrix>, double[]> test = DLDataSet.loadDataSet(Util.readerBZ2(test_path), test_size, inputLayer)
        execute(training.getLeft(), training.getRight(), test.getLeft(), test.getRight());
    }

    void testConvex()
    {
        test_size = 50000
        inputLayer = 784
        outputLayer = 2
        k = 98

        // Convex
        dataset = "Convex";
        training_size = 8000;
        final String training_path = Util.DATAPATH + dataset + "/convex_train.amat.bz2";
        final String test_path = Util.DATAPATH + dataset + "/convex_test.amat.bz2";
        learning_rates = [1e-2, 1e-2, 5e-3, 1e-3, 1e-3, 1e-3]

        Pair<List<DoubleMatrix>, double[]> training = DLDataSet.loadDataSet(Util.readerBZ2(training_path), training_size, inputLayer)
        Pair<List<DoubleMatrix>, double[]> test = DLDataSet.loadDataSet(Util.readerBZ2(test_path), test_size, inputLayer)
        execute(training.getLeft(), training.getRight(), test.getLeft(), test.getRight());
    }

    void construct(final int inputLayer, final int outputLayer)
    {
        // Hidden Layers
        hidden_layers = new ArrayList<>();
        hidden_layers.add(new ReLUNeuronLayer(inputLayer, hiddenLayers[0], L2_Lambda));
        for(int idx = 0; idx < hiddenLayers.length-1; ++idx)
        {
            hidden_layers.add(new ReLUNeuronLayer(hiddenLayers[idx], hiddenLayers[idx+1], L2_Lambda));
        }

        // Output Layers
        NN_layers = new ArrayList<>();
        NN_layers.addAll(hidden_layers);
        NN_layers.add(new SoftMaxNeuronLayer(hiddenLayers[hiddenLayers.length-1], outputLayer, L2_Lambda));
    }

    void execute(List<DoubleMatrix> training_data, double[] training_labels, List<DoubleMatrix> test_data, double[] test_labels)
    {
        assert(size_limits.length == learning_rates.length)

        for(int size = min_layers; size <= max_layers; ++size)
        {
            hiddenLayers = new int[size]
            Arrays.fill(hiddenLayers, hidden_layer_size)

            int[] sum_pool = new int[size]
            Arrays.fill(sum_pool, hidden_pool_size)
            sum_pool[0] = k

            int[] bits = new int[size]
            Arrays.fill(bits, b)

            int[] tables = new int[size]
            Arrays.fill(tables, L)

            for(int idx = 0; idx < size_limits.length; ++idx)
            {
                double[] sl = new double[size]
                Arrays.fill(sl, size_limits[idx])

                System.out.println(make_title())
                construct(inputLayer, outputLayer)
                NN_parameters parameters
                try
                {
                    parameters = new NN_parameters(Util.readerBZ2(Util.DATAPATH + dataset + "/" + Util.MODEL + title), NN_layers, sum_pool, bits, tables, learning_rates[idx], sl)
                }
                catch (Exception ignore)
                {
                    parameters = new NN_parameters(NN_layers, sum_pool, bits, tables, learning_rates[idx], sl)
                }
                NeuralCoordinator NN = new NeuralCoordinator(Double.toString(size_limits[idx]), title, dataset, parameters, NN_layers, hidden_layers, L2_Lambda, new CrossEntropy())
                long startTime = System.currentTimeMillis()
                NN.train(max_epoch, training_data, training_labels, test_data, test_labels)
                long estimatedTime = (System.currentTimeMillis() - startTime) / 1000
                System.out.println(estimatedTime)
            }
        }
    }
}
