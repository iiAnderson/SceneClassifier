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
        if (image.getWidth() != image.getHeight()) {

            if (image.getHeight() > image.getWidth()) {
                newImage = image.extractCenter(image.getWidth(), image.getWidth()).normalise();
                resize.processImage(newImage);
            } else if (image.getWidth() > image.getHeight()) {
                newImage = image.extractCenter(image.getHeight(), image.getHeight()).normalise();
                resize.processImage(newImage);
            }
        } else {
            newImage = image.normalise();
            resize.processImage(image);
        }

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
