package dl.lsh;

import org.jblas.DoubleMatrix;

public interface LSH
{
    int[] hashSignature(DoubleMatrix data);
}
