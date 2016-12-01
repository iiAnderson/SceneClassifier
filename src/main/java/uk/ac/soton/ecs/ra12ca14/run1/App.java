package uk.ac.soton.ecs.ra12ca14.run1;

import org.apache.commons.vfs2.*;
import org.apache.log4j.*;
import org.openimaj.data.dataset.*;
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


        VFSListDataset<FImage> testing = null;
        try {
            testing = new VFSListDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip",
                    ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            Logger.getLogger(App.class).error("Could not read the dataset " + e);
        }

        KNNAnnotator<FImage, String, FloatFV> annotator = KNNAnnotator.create(new OurExtractor(16),
                FloatFVComparison.EUCLIDEAN);

        annotator.train(training);

        for(String ann: annotator.getAnnotations()){
            System.out.println(ann);
        }

        System.out.println("-------------------------");
        int i = 0;
        for(FImage f: testing){
            for(ScoredAnnotation<String> obj: annotator.annotate(f)){
                System.out.println("image " + i + " " +obj.annotation);
            }
            i++;
        }
    }
}
