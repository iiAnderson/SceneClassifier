package uk.ac.soton.ecs.ra12ca14.run3;

import de.bwaldvogel.liblinear.*;
import org.apache.commons.vfs2.*;
import org.apache.log4j.*;
import org.openimaj.data.*;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.*;
import org.openimaj.experiment.dataset.split.*;
import org.openimaj.experiment.evaluation.classification.*;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.*;
import org.openimaj.feature.*;
import org.openimaj.feature.local.data.*;
import org.openimaj.feature.local.list.*;
import org.openimaj.image.*;
import org.openimaj.image.annotation.evaluation.datasets.*;
import org.openimaj.image.feature.dense.gradient.dsift.*;
import org.openimaj.io.*;
import org.openimaj.ml.annotation.linear.*;
import org.openimaj.ml.clustering.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.ml.clustering.kmeans.*;
import org.openimaj.ml.kernel.*;
import org.openimaj.util.pair.*;

import java.io.*;
import java.util.*;

/**
 * OpenIMAJ Tutorial 12 - Chloe Allan
 * Tutorial code to build and train an image classifier to use on the Caltech 101 data set of images. The class
 * constructs a DenseSIFT extractor and uses it to create a PyramidDenseSIFT extractor along with the window
 * size to apply the extractor to. The code also builds a HardAssigner that's used to assign features to identifiers.
 * Exercises 1, 2 and 3 are within the code
 */
public class App {
    public static void main( String[] args ) {
        VFSGroupDataset<FImage> training = null;
        try {
            training = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip",
                    ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            Logger.getLogger(App.class).error("Could not read the dataset " + e);
        }

        training.remove("training");

        VFSListDataset<FImage> testing = null;
        try {
            testing = new VFSListDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip",
                    ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            Logger.getLogger(App.class).error("Could not read the dataset " + e);
        }

        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(training, 80, 10, 10);

        caching(splitter.getTrainingDataset(), testing);
    }

    public static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift)
    {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (FImage rec : sample) {
            FImage img = rec.getImage();

            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);


        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    /**
     * Exercise 2: read/writing the hard assigner to a file and creating a disk caching feature extractor on top of
     * the existing feature extractor
     */
    private static void caching(GroupedDataset<String, ListDataset<FImage>, FImage> data,
                                ListDataset<FImage> testing){

        DenseSIFT dsift = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);

        HardAssigner<byte[], float[], IntFloatPair> assigner =
            trainQuantiser(GroupedUniformRandomisedSampler.sample(data, 30), pdsift);

        FeatureExtractor<DoubleFV, FImage> extractor2 = new PHOWExtractor(pdsift, assigner);

        HomogeneousKernelMap map = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2,
                HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor extractor =  map.createWrappedExtractor(extractor2);

        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L2R_L2LOSS_SVC, 15.0, 0.00001d);

        System.out.println("Started training");
        ann.train(data);
        System.out.println("Finished Training");

        for(int i = 0; i < testing.size(); i ++){
            FImage img = testing.get(i);

            ClassificationResult<String> res = ann.classify(img);

            String app = "";
            for(String s: res.getPredictedClasses())
                app += s;

            System.out.println("Image " + i + " predicted as: " + app);

        }

    }
}
