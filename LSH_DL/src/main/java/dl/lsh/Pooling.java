package dl.lsh;

import org.jblas.DoubleMatrix;

public class Pooling
{
    public static DoubleMatrix compress(final int size, DoubleMatrix data)
    {
        int compressSize = data.length / size;

        DoubleMatrix compressData = DoubleMatrix.zeros(compressSize);
        for(int idx = 0; idx < compressSize; ++idx)
        {
            int offset = idx * size;
            compressData.put(idx, sum(data, offset, offset + size));
        }
        return compressData;
    }

    private static double sum(DoubleMatrix data, int start, int end)
    {
        double value = 0;
        for(int idx = start; idx < end; ++idx)
        {
            value += data.get(idx);
        }
        return value / (end - start);
    }
}
