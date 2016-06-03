package dl.dataset;

import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.List;

public class DLDataSet
{
    public static Pair<List<DoubleMatrix>, double[]> loadDataSet(BufferedReader stream, final int size, final int numAttributes) throws Exception
    {
        // read LibSVM data
        double[] label_list = new double[size];
        List<DoubleMatrix> data_list = new ArrayList<>(size);

        for(int label_idx = 0; label_idx < size; ++label_idx)
        {
            String[] data = stream.readLine().trim().split("\\s+");
            label_list[label_idx] = Double.parseDouble(data[numAttributes]);

            DoubleMatrix feature_vector = DoubleMatrix.zeros(numAttributes);
            for (int idx = 0; idx < numAttributes; ++idx)
            {
                feature_vector.put(idx, Double.parseDouble(data[idx]));
                assert(feature_vector.get(idx) >= 0.0 && feature_vector.get(idx) <= 1.0);
            }
            data_list.add(feature_vector);
        }
        return new DMPair(data_list, label_list);
    }
}
