package dl.nn;

public interface ICostFunction
{
    double correct(double[] y_hat, double labels);
    double accuracy(double[][] y_hat, double[] labels);
    double costFunction(double[] y_hat, double labels);
    double[] outputDelta(double[] y_hat, double labels, NeuronLayer l);
}
