package uk.ac.soton.ecs.ra12ca14.run1;

import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.image.processing.resize.*;

/**
 * Created by chloeallan on 30/11/2016.
 */
public class OurExtractor implements FeatureExtractor<DoubleFV, FImage> {

    private ResizeProcessor resize = null;

    public OurExtractor(int size){
        resize = new ResizeProcessor(size, size);
    }

    public DoubleFV extractFeature(FImage image) {
        FImage newImage = null;

        int[] vals = image.getHeight() > image.getWidth() ?
                new int[]{image.getWidth(), image.getHeight()} :
                new int[]{image.getHeight(), image.getWidth()};

        if (image.getWidth() != image.getHeight()) {

            newImage = image.extractCenter(vals[0], vals[1]);
            resize.processImage(newImage);
        } else
            newImage = image;

        resize.processImage(newImage);

        float tot = 0;
        int pixelTot = 0;
        for(int i = 0; i < newImage.getWidth(); i++){
            for(int j = 0; j < newImage.getHeight(); j++){
                tot += newImage.pixels[j][i];
                pixelTot++;
            }
        }

        return new FloatFV(newImage.subtractInplace(tot/pixelTot)
                .getFloatPixelVector()).normaliseFV();
    }
}
