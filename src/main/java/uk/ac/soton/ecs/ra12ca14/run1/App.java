package uk.ac.soton.ecs.ra12ca14.run1;

import org.apache.commons.vfs2.*;
import org.apache.log4j.*;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.split.*;
import org.openimaj.experiment.evaluation.classification.*;
import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.ml.annotation.*;
import org.openimaj.ml.annotation.basic.*;

import java.util.*;

/**
 * Created by chloeallan on 30/11/2016.
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

        KNNAnnotator<FImage, String, FloatFV> annotator = KNNAnnotator.create(new OurExtractor(16),
                FloatFVComparison.EUCLIDEAN);


        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(training, 50, 0, 50);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSplit = splitter.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testingSplit = splitter.getTestDataset();

        annotator.trainMultiClass(trainingSplit);

        int i = 0;
        for(FImage image: testingSplit) {
            if (i > 30) {
                break;
            }
            FImage newImage = null;
            if (image.getHeight() > image.getWidth()) {
                newImage = image.extractCenter(image.getWidth(), image.getWidth()).normalise();
            } else if (image.getWidth() > image.getHeight()) {
                newImage = image.extractCenter(image.getHeight(), image.getHeight()).normalise();
            }
            DisplayUtilities.display(newImage);
            i++;
        }
    }
}
