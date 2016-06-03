package dl.lsh;

import org.jblas.DoubleMatrix;
import java.util.List;

public class RandomProjection
{
    private List<DoubleMatrix> m_projection_matrix;
    private DoubleMatrix m_query;
    private int[] m_hashes;

    public RandomProjection(int[] hashes, List<DoubleMatrix> projection_matrix, DoubleMatrix query)
    {
        m_projection_matrix = projection_matrix;
        m_query = query;
        m_hashes = hashes;
    }

    public int[] run()
    {
        int hash_idx = -1;
        for(DoubleMatrix projection : m_projection_matrix)
        {
            assert(projection.columns == m_query.rows);
            DoubleMatrix dotProduct = projection.mmul(m_query);

            int signature = 0;
            for(int idx = 0; idx < dotProduct.length; ++idx)
            {
                signature |= sign(dotProduct.get(idx));
                signature <<= 1;
            }
            m_hashes[++hash_idx] = signature;
        }
        return m_hashes;
    }

    private int sign(double value)
    {
        return (value >= 0) ? 1 : 0;
    }
}
