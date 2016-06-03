package dl.lsh;

import org.jblas.DoubleMatrix;

import java.util.*;

public class HashBuckets
{
    private double m_nn_sizeLimit;
	private int m_L;
    private int m_poolDim;
    private LSH m_hashFunction;
	private List<HashMap<Integer, Set<Integer>>> m_Tables = new ArrayList<>();
    private List<HashMap<Integer, int[]>> m_bucket_hashes = new ArrayList<>();

	public HashBuckets(double sizeLimit, int poolDim, int L, LSH hashFunction)
    {
        m_hashFunction = hashFunction;
        m_poolDim = poolDim;
        m_nn_sizeLimit = sizeLimit;
		m_L = L;
        construct();
	}

    public void construct()
    {
        for (int i = 0; i < m_L; i++)
        {
            m_Tables.add(new HashMap<>());
            m_bucket_hashes.add(new HashMap<>());
        }
    }

    public void clear()
    {
        m_Tables.clear();
        m_bucket_hashes.clear();
        construct();
    }

    public void LSHAdd(int recIndex, DoubleMatrix data)
    {
        LSHAdd(recIndex, generateHashSignature(data));
    }

	private void LSHAdd(int recIndex, int[] hashes)
    {
        assert(hashes.length == m_L);

		for (int idx = 0; idx < m_L; idx++)
        {
            if (!m_Tables.get(idx).containsKey(hashes[idx]))
            {
                Set<Integer> set = new HashSet<>();
                set.add(recIndex);
                m_Tables.get(idx).put(hashes[idx], set);
                m_bucket_hashes.get(idx).put(hashes[idx], hashes);
            }
            else
            {
                m_Tables.get(idx).get(hashes[idx]).add(recIndex);
            }
		}
	}

    public Set<Integer> LSHUnion(DoubleMatrix data)
    {
        return LSHUnion(generateHashSignature(data));
    }

    public Set<Integer> histogramLSH(DoubleMatrix data)
    {
        return histogramLSH(generateHashSignature(data));
    }

    public Set<Integer> histogramLSH(int[] hashes)
    {
        assert(hashes.length == m_L);

        Histogram hist = new Histogram();
        for (int idx = 0; idx < m_L; ++idx)
        {
            if (m_Tables.get(idx).containsKey(hashes[idx]))
            {
                hist.add(m_Tables.get(idx).get(hashes[idx]));
            }
        }
        return hist.thresholdSet(m_nn_sizeLimit);
    }

	public Set<Integer> LSHUnion(int[] hashes)
    {
        assert(hashes.length == m_L);

        Set<Integer> retrieved = new HashSet<>();
		for (int idx = 0; idx < m_L && retrieved.size() < m_nn_sizeLimit; ++idx)
        {
			if (m_Tables.get(idx).containsKey(hashes[idx]))
            {
                retrieved.addAll(m_Tables.get(idx).get(hashes[idx]));
            }
		}
		return retrieved;
	}

    public int[] generateHashSignature(DoubleMatrix data)
    {
        return m_hashFunction.hashSignature(Pooling.compress(m_poolDim, data));
    }
}
