package uk.ac.soton.ecs.ra12ca14.run2;

import org.openimaj.feature.*;
import org.openimaj.feature.local.*;
import org.openimaj.feature.local.list.*;
import org.openimaj.image.*;
import org.openimaj.image.feature.local.aggregate.*;
import org.openimaj.image.pixel.sampling.*;
import org.openimaj.math.geometry.point.Point2d;
import org.openimaj.math.geometry.shape.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.util.pair.*;

import java.util.*;

public class OurExtractor implements FeatureExtractor<DoubleFV, FImage> {

    HardAssigner<float[], float[], IntFloatPair> assigner;

    public OurExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
        this.assigner = assigner;
    }

    public DoubleFV extractFeature(FImage image) {

        BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(assigner);

        //mean center image
        float tot = 0;
        int pixelTot = 0;
        for(int i = 0; i < image.getWidth(); i++){
            for(int j = 0; j < image.getHeight(); j++){
                tot += image.pixels[j][i];
                pixelTot++;
            }
        }

        RectangleSampler sampler = new RectangleSampler(image.subtractInplace(tot/pixelTot).normalise(), 4, 4, 8, 8);

        Iterator<Rectangle> iterator = sampler.iterator();
        LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> list =
                new MemoryLocalFeatureList<>();

        pullLocalFeaturesRectangle(iterator, image, list);

        return bovw.aggregate(list).normaliseFV();
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
}