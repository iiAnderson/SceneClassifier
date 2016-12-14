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

import java.io.*;
import java.util.*;

/**
 * Run 1: KNN Tiny Image Classifier
 */
public class App {

    /*
        Creates the KNN annotator and passes it a copy of the Extractor.
        Then Splits the data and trains the classifier on the training data.
     */
    public static void main(String[] args) {

        VFSGroupDataset<FImage> training = null;
        try {
            training = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip!/training",
                    ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            Logger.getLogger(App.class).error("Could not read the dataset " + e);
            return;
        }


        VFSListDataset<FImage> testing = null;
        try {
            testing = new VFSListDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip",
                    ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            Logger.getLogger(App.class).error("Could not read the dataset " + e);
            return;
        }

        KNNAnnotator<FImage, String, DoubleFV> annotator = KNNAnnotator.create(new TinyImageExtractor(16),
                DoubleFVComparison.EUCLIDEAN);


        //Splits the data into training, validation and testing
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(training, 60, 20, 20);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSplit = splitter.getTrainingDataset();

        //Trains the annotator
        annotator.trainMultiClass(trainingSplit);

        //File outputting.
        File output = new File("run1.txt");
        FileWriter fileWriter = null;
        try {
            if(!output.exists())
                output.createNewFile();
        }catch (Exception e){}

        try {
            fileWriter = new FileWriter(output);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        PrintWriter printer = new PrintWriter(fileWriter);

        for(int i = 0; i < testing.size(); i ++){
            FileObject img = testing.getFileObject(i);

            ClassificationResult<String> res = annotator.classify(testing.get(i));

            String app = "";
            for(String s: res.getPredictedClasses())
                app += s;

            String out = img.getName().getBaseName() + " " + app;
            System.out.println(out);
            printer.println(out);

        }
    }
}
