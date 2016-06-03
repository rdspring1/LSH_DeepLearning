package dl.nn;

import java.util.Arrays;

// Compare Against Labels - Classification
public class CrossEntropy implements ICostFunction
{
    private int max_idx(double[] array)
    {
        int max_idx = 0;
        double max_value = Double.MIN_VALUE;
        for(int idx = 0; idx < array.length; ++idx)
        {
            if(max_value < array[idx])
            {
                max_idx = idx;
                max_value = array[idx];
            }
        }
        return max_idx;
    }

    public double correct(double[] y_hat, double labels)
    {
        return (max_idx(y_hat) == (int) labels) ? 1.0 : 0.0;
    }

    public double accuracy(double[][] y_hat, double[] labels)
    {
        // select highest probability index as label for data set
        // check for matches and return average
        double correct = 0;
        for(int idx = 0; idx < labels.length; ++idx)
        {
            if(max_idx(y_hat[idx]) == (int) labels[idx])
            {
                ++correct;
            }
        }
        return correct / labels.length;
    }

    public double costFunction(double[] y_hat, double labels)
    {
        return -Math.log(y_hat[(int) labels]);
    }

    public double[] outputDelta(double[] y_hat, double labels, NeuronLayer l)
    {
        double[] delta = Arrays.copyOf(y_hat, y_hat.length);
        delta[(int) labels] -= 1.0;
        return delta;
    }
}
