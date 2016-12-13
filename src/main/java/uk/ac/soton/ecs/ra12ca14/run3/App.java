package uk.ac.soton.ecs.ra12ca14.run3;

import org.apache.commons.vfs2.*;
import org.apache.log4j.*;
import org.openimaj.data.*;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.*;
import org.openimaj.experiment.dataset.split.*;
import org.openimaj.experiment.evaluation.classification.*;
import org.openimaj.feature.*;
import org.openimaj.feature.local.data.*;
import org.openimaj.feature.local.list.*;
import org.openimaj.image.*;
import org.openimaj.image.feature.dense.gradient.dsift.*;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.ml.clustering.kmeans.*;
import org.openimaj.util.pair.*;

import java.io.*;
import java.util.*;

public class App {
    public static void main( String[] args ) {
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

        GroupedRandomSplitter<String, FImage> splitter =
                new GroupedRandomSplitter<>(training, 80, 10, 10);

        performTask(splitter.getTrainingDataset(), testing);
    }

    public static HardAssigner<byte[], float[], IntFloatPair> trainWithKMeans(
            Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (FImage rec : sample) {
            FImage img = rec.getImage();

            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        //Takes the first 10000 keys
        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);


        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    private static void performTask(GroupedDataset<String, ListDataset<FImage>, FImage> data,
                                VFSListDataset<FImage> testing){

        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(
                new DenseSIFT(5, 7), 6f, 7);

        //Build the Vocab and save it in the HardAssigner
        HardAssigner<byte[], float[], IntFloatPair> assigner =
                trainWithKMeans(GroupedUniformRandomisedSampler.sample(data, 30), pdsift);

        FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);

        NaiveBayesAnnotator<FImage, String> ann = new NaiveBayesAnnotator<>(
                extractor, NaiveBayesAnnotator.Mode.ALL);

        System.out.println("Started training");

        //Training the data
        ann.train(data);

        System.out.println("Finished Training");

        File output = new File("run3.txt");
        try {
            output.createNewFile();
        } catch(Exception e){}

        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter(output);
        } catch (IOException e) {
            e.printStackTrace();
            return;
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

            //Classifies the Image and returns a result
            ClassificationResult<String> res = ann.classify((FImage) content);

            String app = "";
            for(String s: res.getPredictedClasses())
                app += s;

            //Prints out to file and System
            String out = "Image " + img.getName() + " predicted as: " + app;
            System.out.println(out);
            printer.println(out);

        }

    }
}
