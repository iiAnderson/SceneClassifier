package uk.ac.soton.ecs.ra12ca14.run1;

import org.apache.commons.vfs2.*;
import org.apache.log4j.*;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.split.*;
import org.openimaj.experiment.evaluation.classification.*;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.*;
import org.openimaj.feature.*;
import org.openimaj.image.*;
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

        KNNAnnotator<FImage, String, DoubleFV> annotator = KNNAnnotator.create(new OurExtractor(16),
                DoubleFVComparison.EUCLIDEAN);


        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(training, 50, 0, 50);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSplit = splitter.getTrainingDataset();

        annotator.trainMultiClass(trainingSplit);


        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<>(
                        annotator, splitter.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result);
    }
}
