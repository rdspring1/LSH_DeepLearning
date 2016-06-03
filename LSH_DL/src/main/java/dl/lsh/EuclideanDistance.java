package dl.lsh;

import org.jblas.DoubleMatrix;

public class EuclideanDistance
{
    public static double distance(DoubleMatrix x, DoubleMatrix y)
    {
        return x.distance2(y);
    }
    
    public static double distance(double[] x, double[] y)
    {
        assert(x.length == y.length);
        double distance = 0.0;
        double x_mag = 0;
        double y_mag = 0;

        for(int idx = 0; idx < x.length; ++idx)
        {
            x_mag += Math.pow(x[idx], 2.0);
            y_mag += Math.pow(y[idx], 2.0);
        }
        x_mag = Math.sqrt(x_mag);
        y_mag = Math.sqrt(y_mag);

        for(int idx = 0; idx < x.length; ++idx)
        {
            distance += Math.pow((x[idx] / x_mag) - (y[idx] / y_mag), 2.0);
        }
        return Math.sqrt(distance);
    }
}
