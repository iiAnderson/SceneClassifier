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

public class RectangleSamplerExtractor implements FeatureExtractor<DoubleFV, FImage> {

    //Contains the vocab
    HardAssigner<float[], float[], IntFloatPair> assigner;

    public RectangleSamplerExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
        this.assigner = assigner;
    }

    /*
        Takes the input image and mean centers it to improve classification, then uses a rectanglesampler to
        iterate through the image, creating sub-images from the rectangles and pulling the feature vectors
        from these sub images.
        These featurevectors are then aggregated by the BoVW to produce a sparsevector, which is normalised into
        a DoubleFV and returned
     */
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


        //Iterates through the rectangles
        while(iterator.hasNext()) {

            Rectangle rec = iterator.next();
            Point2d center = rec.calculateCentroid();

            list.add(new LocalFeatureImpl<>(
                    new SpatialLocation(center.getX(), center.getY()),
                    new FloatFV(image.extractROI(rec).getFloatPixelVector())));
        }

        return bovw.aggregate(list).normaliseFV();
    }
}