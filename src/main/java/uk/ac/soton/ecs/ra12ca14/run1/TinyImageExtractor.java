package uk.ac.soton.ecs.ra12ca14.run1;

import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.image.processing.resize.*;

/**
 * TinyImageExtractor takes the input image and outputs a feature vector of the image in tiny image form.
 */
public class TinyImageExtractor implements FeatureExtractor<DoubleFV, FImage> {

    //Used in resizing the image.
    private ResizeProcessor resize = null;

    public TinyImageExtractor(int size){
        resize = new ResizeProcessor(size, size);
    }

    /*
        This method squares the image by extracting the center of the image according to the shortest length
        and then resizes the image to 16x16.

        Then returns the Normalised (to unit length) DoubleFV.
     */
    public DoubleFV extractFeature(FImage image) {
        FImage newImage = null;


        if (image.getWidth() != image.getHeight()) {

            if (image.getHeight() > image.getWidth()) {
                newImage = image.extractCenter(image.getWidth(), image.getWidth());
                resize.processImage(newImage);
            } else if (image.getWidth() > image.getHeight()) {
                newImage = image.extractCenter(image.getHeight(), image.getHeight());
                resize.processImage(newImage);
            }
        } else {
            newImage = image.normalise();
            resize.processImage(newImage);
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
