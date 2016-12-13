package uk.ac.soton.ecs.ra12ca14.run2;

import de.bwaldvogel.liblinear.*;
import org.apache.commons.vfs2.*;
import org.apache.log4j.*;
import org.bridj.cpp.std.*;
import org.openimaj.data.*;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.*;
import org.openimaj.experiment.dataset.split.*;
import org.openimaj.experiment.evaluation.classification.*;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.*;
import org.openimaj.feature.*;
import org.openimaj.feature.local.*;
import org.openimaj.feature.local.data.*;
import org.openimaj.feature.local.list.*;
import org.openimaj.image.*;
import org.openimaj.image.feature.dense.gradient.dsift.*;
import org.openimaj.image.pixel.sampling.*;
import org.openimaj.math.geometry.point.*;
import org.openimaj.math.geometry.shape.*;
import org.openimaj.ml.annotation.linear.*;
import org.openimaj.ml.clustering.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.ml.clustering.kmeans.*;
import org.openimaj.util.pair.*;

import java.util.*;

import static uk.ac.soton.ecs.ra12ca14.run2.App.pullLocalFeaturesRectangle;

/**
 * Created by chloeallan on 09/12/2016.
 */
public class App {

    public static void main(String[] args) {

        VFSGroupDataset<FImage> training = null;
        try {
            training = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip!/training",
                    ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            Logger.getLogger(App.class).error("Could not read the dataset " + e);
            return;
        }

        VFSListDataset<FImage> testing = null;
        try {
            testing = new VFSListDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip",
                    ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            Logger.getLogger(App.class).error("Could not read the dataset " + e);
        }

        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(training, 80, 20, 0);


        HardAssigner<float[], float[], IntFloatPair> assigner =
                trainWithKMeans(GroupedUniformRandomisedSampler.sample(splitter.getTrainingDataset(), 20));

        System.out.println("Built Extractor");
        OurExtractor extractor = new OurExtractor(assigner);
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<>(extractor,
                LiblinearAnnotator.Mode.MULTILABEL, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001d);

        annotator.train(training);
        System.out.println("Training completed");

        for(int i = 0; i < testing.size(); i ++){
            FImage img = testing.get(i);

            ClassificationResult<String> res = annotator.classify(img);

            String app = "";
            for(String s: res.getPredictedClasses())
                app += s;

            System.out.println("Image " + i + " predicted as: " + app);

        }
    }

    static void pullLocalFeaturesRectangle(Iterator<Rectangle> iterator, FImage image,
                                              LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> features){
        while(iterator.hasNext()) {

            Rectangle rec = iterator.next();
            Point2d center = rec.calculateCentroid();

            features.add(new LocalFeatureImpl<>(
                    new SpatialLocation(center.getX(), center.getY()),
                    new FloatFV(image.extractROI(rec).getFloatPixelVector())));
        }
    }

    private static HardAssigner<float[], float[], IntFloatPair> trainWithKMeans(
            Dataset<FImage> sample) {

        LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> features =
                new MemoryLocalFeatureList<>();

        List<FloatFV> vec = new ArrayList<>();

        for (FImage image : sample) {
            RectangleSampler sampler = new RectangleSampler(image, 4, 4, 8, 8);

            Iterator<Rectangle> iterator = sampler.iterator();

            while(iterator.hasNext()) {

                Rectangle rec = iterator.next();

                vec.add(new FloatFV(image.extractROI(rec).getFloatPixelVector()));
            }
        }

        System.out.println(features.size());

        float[][] vectors = new float[10000][];

        for(int i = 0; i < 10000; i++){
            vectors[i] = vec.get(i).getVector();
        }


        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(300);

        FloatCentroidsResult result = km.cluster(vectors);

        return result.defaultHardAssigner();
    }
}
