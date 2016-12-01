package uk.ac.soton.ecs.ra12ca14.run1;

import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.image.processing.resize.*;

/**
 * Created by chloeallan on 30/11/2016.
 */
public class OurExtractor implements FeatureExtractor<FloatFV, FImage> {

    private ResizeProcessor resize = null;

    public OurExtractor(int size){
        resize = new ResizeProcessor(size, size);
    }


    public FloatFV extractFeature(FImage image) {
        FImage newImage = null;
        if (image.getWidth() != image.getHeight()) {

            if (image.getHeight() > image.getWidth()) {
                int newy = (image.getHeight() - image.getWidth()) / 2;
                newImage = image.extractROI(0, newy, image.getWidth(), image.getWidth());
                resize.processImage(newImage);
            } else if (image.getWidth() > image.getHeight()) {
                int newx = (image.getWidth() - image.getHeight()) / 2;
                newImage = image.extractROI(newx, 0, image.getHeight(), image.getHeight());
                resize.processImage(newImage);
            }
        } else {
            newImage = image;
            resize.processImage(image);

        }
        return new FloatFV(newImage.getFloatPixelVector());
    }
}
