package dl.dataset;

import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

public class MNISTDataSet
{
    private static final int LABEL_MAGIC = 2049;
    private static final int IMAGE_MAGIC = 2051;

    public static Pair<List<DoubleMatrix>, double[]> loadDataSet(final String label_path, final String image_path) throws Exception
    {
        // read MNIST data
        DataInputStream label_stream = new DataInputStream(new FileInputStream(label_path));
        DataInputStream image_stream = new DataInputStream(new FileInputStream(image_path));

        int label_magicNumber = label_stream.readInt();
        if (label_magicNumber != LABEL_MAGIC)
        {
            System.err.println("Label file has wrong magic number: " + label_magicNumber + " expected: " + LABEL_MAGIC);
        }
        int image_magicNumber = image_stream.readInt();
        if (image_magicNumber != IMAGE_MAGIC)
        {
            System.err.println("Image file has wrong magic number: " + label_magicNumber + " expected: " + IMAGE_MAGIC);
        }

        int numLabels = label_stream.readInt();
        int numImages = image_stream.readInt();
        int numRows = image_stream.readInt();
        int numCols = image_stream.readInt();
        if (numLabels != numImages)
        {
            System.err.println("Image file and label file do not contain the same number of entries.");
            System.err.println("  Label file contains: " + numLabels);
            System.err.println("  Image file contains: " + numImages);
        }

        int label_idx = 0;
        int numImagesRead = 0;
        double[] label_list = new double[numLabels];
        List<DoubleMatrix> image_list = new ArrayList<>(numImages);
        while (label_stream.available() > 0 && label_idx < numLabels)
        {
            DoubleMatrix image = DoubleMatrix.zeros(numCols * numRows);
            label_list[label_idx++] = label_stream.readByte();
            int image_idx = 0;
            for (int colIdx = 0; colIdx < numCols; colIdx++)
            {
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
                {
                    image.put(image_idx++, image_stream.readUnsignedByte() / 255.0);
                }
            }
            image_list.add(image);
            ++numImagesRead;
        }
        assert(label_idx == numImagesRead);
        return new DMPair(image_list, label_list);
    }
}
