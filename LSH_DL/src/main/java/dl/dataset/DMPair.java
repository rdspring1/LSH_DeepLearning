package dl.dataset;

import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;
import java.util.List;

public class DMPair extends Pair<List<DoubleMatrix>, double[]>
{
    private List<DoubleMatrix> m_left;
    private double[] m_right;
    public DMPair(List<DoubleMatrix> left, double[] right)
    {
        m_left = left;
        m_right = right;
    }

    @Override
    public List<DoubleMatrix> getLeft()
    {
        return m_left;
    }

    @Override
    public double[] getRight()
    {
        return m_right;
    }

    public double[] setValue(double[] value)
    {
        return null;
    }
}


