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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * Created by chloeallan on 30/11/2016.
 */
public class App {

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

        KNNAnnotator<FImage, String, DoubleFV> annotator = KNNAnnotator.create(new OurExtractor(16),
                DoubleFVComparison.EUCLIDEAN);


        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(training, 50, 0, 50);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSplit = splitter.getTrainingDataset();

        annotator.trainMultiClass(trainingSplit);

        File output = new File("./output/run1.txt");
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter(output);
        } catch (IOException e) {
            e.printStackTrace();
        }
        PrintWriter printer = new PrintWriter(fileWriter);

        for(int i = 0; i < testing.size(); i ++){
            FileObject img = testing.getFileObject(i);
            FileContent content = null;
            try {
                content = img.getContent();
            } catch (FileSystemException e) {
                e.printStackTrace();
                return;
            }

            ClassificationResult<String> res = annotator.classify((FImage) content);

            String app = "";
            for(String s: res.getPredictedClasses())
                app += s;

            String out = "Image " + img.getName() + " predicted as: " + app;
            System.out.println(out);
            printer.println(out);

        }
    }
}
