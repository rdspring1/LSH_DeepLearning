package dl.lsh;

import dl.nn.Util;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class CosineDistance implements LSH
{
    private int m_L;
    private List<DoubleMatrix> randomMatrix;
    private int[] hashes;

    public CosineDistance(int b, int L, int d)
    {
        m_L = L;
        randomMatrix = new ArrayList<>();
        hashes = new int[m_L];
        for(int jdx = 0; jdx < m_L; ++jdx)
        {
            randomMatrix.add(DoubleMatrix.randn(b, d));
        }
    }

    public int[] hashSignature(DoubleMatrix data)
    {
        return new RandomProjection(hashes, randomMatrix, data).run();
    }

    public static double distance(DoubleMatrix x, DoubleMatrix y)
    {
        return 1 - (x.dot(y) / (x.norm2() * y.norm2()));
    }

    public static double distance(double[] x, double[] y)
    {
        assert(x.length == y.length);
        double dp = 0.0;
        double x_norm = 0.0;
        double y_norm = 0.0;

        for(int idx = 0; idx < x.length; ++idx)
        {
            dp += x[idx] * y[idx];
            x_norm += Math.pow(x[idx], 2);
            y_norm += Math.pow(y[idx], 2);
        }

        x_norm = Math.sqrt(x_norm);
        y_norm = Math.sqrt(y_norm);
        return 1 - (dp / (x_norm * y_norm));
    }

    public static double dotProductDistance(int[] x, int[] y, final int b)
    {
        final int numIntegers = b / Util.INT_SIZE;
        assert(x.length == numIntegers);
        assert(y.length == numIntegers);

        double dp = 0.0;
        double x_norm = 0.0;
        double y_norm = 0.0;

        for(int idx = 0; idx < x.length; ++idx)
        {
            dp += count(x[idx] & y[idx]);
            x_norm += count(x[idx]);
            y_norm += count(y[idx]);
        }

        x_norm = Math.sqrt(x_norm);
        y_norm = Math.sqrt(y_norm);
        return 1 - (dp / (x_norm * y_norm));
    }

    public static double hammingDistance(int[] x, int[] y, final int b)
    {
        final int numIntegers = b / Util.INT_SIZE;
        int numBits = b % Util.INT_SIZE;
        numBits = (numBits == 0) ? Util.INT_SIZE : numBits;
        final int bitMask = (int) Math.pow(2, numBits) - 1;
        assert(x.length == numIntegers);
        assert(y.length == numIntegers);

        int hammingDistance = 0;
        for(int idx = 0; idx < x.length-1; ++idx)
        {
            hammingDistance += count(x[idx] ^ y[idx]);
        }

        hammingDistance += count((x[x.length-1] & bitMask) ^ (y[y.length-1] & bitMask));
        return 1 - Math.cos((double) hammingDistance * Math.PI / (double) b);
    }

    private static int count(int value)
    {
        int count = 0;
        for(int idx = 0; idx < Util.INT_SIZE; ++idx)
        {
            count += (value & 1);
            value >>= 1;
        }
        return count;
    }
}
