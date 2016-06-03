package dl.nn;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream;
import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.*;

public class Util
{
    public static final String DATAPATH = "../data/";
    public static final String MODEL = "Models/";
    public static final String TRAIN = "Train/";
    public static final String TEST = "Test/";

    public final static int INT_SIZE = 32;
    // ASGD - Higher Threads for larger workloads improves performance
    public final static int LAYER_THREADS = 1;
    public final static int UPDATE_SIZE = 10;

    public static Random rand = new Random(System.currentTimeMillis());
    public static int randInt(int min, int max)
    {
        return rand.nextInt((max - min)) + min;
    }

    public static boolean randBoolean(double probability)
    {
        return rand.nextDouble() < probability;
    }

    public static BufferedWriter writerBZ2(final String path) throws IOException
    {
        return new BufferedWriter(new OutputStreamWriter(new BZip2CompressorOutputStream(new BufferedOutputStream(new FileOutputStream(path)))));
    }

    public static BufferedReader readerBZ2(final String path) throws IOException
    {
        return new BufferedReader(new InputStreamReader(new BZip2CompressorInputStream(new BufferedInputStream(new FileInputStream(path)))));
    }

    public static DataInputStream byteReaderBZ2(final String path) throws Exception
    {

        return new DataInputStream(new BZip2CompressorInputStream(new BufferedInputStream(new FileInputStream(path))));
    }

    public static DoubleMatrix vectorize(double[] data)
    {
        return vectorize(data, 0, data.length);
    }

    public static DoubleMatrix vectorize(double[] data, int offset, int length)
    {
        DoubleMatrix vector = DoubleMatrix.zeros(length);
        for(int idx = 0; idx < length; ++idx)
        {
            vector.put(idx, 0, data[offset + idx]);
        }
        return vector;
    }

    public static List<DoubleMatrix> mean_normalization(double[] sum, List<DoubleMatrix> data_list)
    {
        DoubleMatrix meanVector = new DoubleMatrix(sum);
        meanVector.divi(data_list.size());
        for(DoubleMatrix data : data_list)
        {
            data.subi(meanVector);
        }
        return data_list;
    }

    public static List<DoubleMatrix> range_normalization(double[] min, double[] max, List<DoubleMatrix> data_list)
    {
        DoubleMatrix minVector = new DoubleMatrix(min);
        DoubleMatrix maxVector = new DoubleMatrix(max);
        DoubleMatrix range = maxVector.sub(minVector);
        for(DoubleMatrix data : data_list)
        {
            data.divi(range);
        }
        return data_list;
    }

    public static double gradient_check(NeuralNetwork NN, List<DoubleMatrix> data, double[] labels, int num_checks)
    {
        List<int[]> input_hashes = NN.computeHashes(data);
        final double delta = 0.0001;

        double max = 0.0;
        double[] original_params = NN.copyTheta();
        for(int n = 0; n < num_checks; ++n)
        {
            int randData = randInt(0, labels.length);
            int randIdx = randInt(0, NN.numTheta());
            double theta = NN.getTheta(randIdx);

            NN.execute(input_hashes.get(randData), data.get(randData), labels[randData], false);
            double gradient = NN.getGradient(randIdx);
            NN.setTheta(original_params);

            NN.setTheta(randIdx, theta-delta);
            NN.execute(input_hashes.get(randData), data.get(randData), labels[randData], false);
            double J0 = NN.getCost();
            NN.setTheta(original_params);

            NN.setTheta(randIdx, theta+delta);
            NN.execute(input_hashes.get(randData), data.get(randData), labels[randData], false);
            double J1 = NN.getCost();
            NN.setTheta(original_params);

            double est_gradient = (J1-J0) / (2*delta);
            double error = Math.abs(gradient - est_gradient);
            System.out.println("Error: " + error + " Gradient: " + gradient + " Est.Gradient: " + est_gradient);
            max = Math.max(max, error);
        }
        return max;
    }

    public static <E extends Thread> List<E> join(List<E> threads)
    {
        for(E t : threads)
        {
            try
            {
                t.join();
            }
            catch (InterruptedException e)
            {
                System.out.println("Thread interrupted");
            }
        }
        return threads;
    }
}
