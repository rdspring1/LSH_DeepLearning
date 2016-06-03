package dl.lsh;

import org.apache.commons.lang3.mutable.MutableInt;

import java.util.*;

public class Histogram
{
    private HashMap<Integer, MutableInt> histogram = new HashMap<>();

    public void add(Collection<Integer> data)
    {
        for(Integer value : data)
        {
            if(!histogram.containsKey(value))
            {
                histogram.put(value, new MutableInt(1));
            }
            else
            {
                histogram.get(value).increment();
            }
        }
    }

    public Set<Integer> thresholdSet(double count)
    {
        List<Map.Entry<Integer, MutableInt>> list = new LinkedList<>(histogram.entrySet());
        Collections.sort(list, (Map.Entry<Integer,MutableInt> o1, Map.Entry<Integer, MutableInt> o2) -> o2.getValue().compareTo(o1.getValue()));
        count = Math.min(count, list.size());

        Set<Integer> retrieved = new HashSet<>();
        Iterator<Map.Entry<Integer, MutableInt>> iterator = list.iterator();
        for(int idx = 0; idx < count; ++idx)
        {
            retrieved.add(iterator.next().getKey());
        }
        return retrieved;
    }
}
