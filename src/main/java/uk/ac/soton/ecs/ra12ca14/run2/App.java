package uk.ac.soton.ecs.ra12ca14.run2;

import de.bwaldvogel.liblinear.*;
import org.apache.commons.vfs2.*;
import org.apache.log4j.*;
import org.bridj.cpp.std.*;
import org.openimaj.data.*;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.*;
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

/**
 * Created by chloeallan on 09/12/2016.
 */
public class App {

    public static void main(String[] args) {

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

        Map<String, FImage> map = new HashMap<>();

        for(FImage image : training){
            map.put("image", image);
        }
        HardAssigner<float[], float[], IntFloatPair> assigner = null;
        assigner = trainWithKMeans(map);

        OurExtractor extractor = new OurExtractor(assigner);
        LiblinearAnnotator annotator = new LiblinearAnnotator(extractor,
                LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        annotator.train(training);

    }

    static HardAssigner<float[], float[], IntFloatPair> trainWithKMeans(
            Map<String, FImage> sample) {

        LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> features =
                new MemoryLocalFeatureList<>();

        for (Map.Entry<String, FImage> entry : sample.entrySet()) {
            FImage image = entry.getValue();
            RectangleSampler sampler = new RectangleSampler(image, 4, 4, 8, 8);

            Iterator<Rectangle> iterator = sampler.iterator();

            Rectangle rec = iterator.next();
            Point2d center = rec.calculateCentroid();
            float[][] pixels = new float[(int) rec.getWidth()][(int) rec.getHeight()];
            for(int x = (int) rec.minX(); x < rec.getWidth(); x++){
                for(int y = (int) rec.minY(); y < rec.getHeight(); y++){
                    pixels[x][y] = image.pixels[x][y];
                }
            }



            features.add(new LocalFeatureImpl<>(new SpatialLocation(center.getX(), center.getY()),
                    new FloatFV(new FImage(pixels).getFloatPixelVector())));
        }

        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        DataSource<float[]> datasource =
                new LocalFeatureListDataSource<>(features);
        FloatCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }
}
