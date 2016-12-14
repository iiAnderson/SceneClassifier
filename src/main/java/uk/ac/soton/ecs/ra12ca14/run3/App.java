package uk.ac.soton.ecs.ra12ca14.run3;

import de.bwaldvogel.liblinear.SolverType;
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
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.ml.clustering.kmeans.*;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.*;

import java.io.*;
import java.util.*;

public class App {

    /*
        Imports the datasets and splits them, calling perform task.
     */
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
                new GroupedRandomSplitter<>(training, 30, 40, 30);

        performTask(splitter.getTrainingDataset(), testing, splitter.getValidationDataset());
    }

    /*
        Trains the hard assigner by creating a vocab of visual words using KMeans.
        Returns the HardAssigner containing the vocab.
     */
    public static HardAssigner<byte[], float[], IntFloatPair> trainWithKMeans(
            Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift) {

        //local features with location and vector
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (FImage rec : sample) {
            FImage img = rec.getImage();

            pdsift.analyseImage(img);

            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        //Takes the first 10000 keys
        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);


        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        //hard assigner containing the vocab
        return result.defaultHardAssigner();
    }

    /*
        Performs the classifier. First gets the Hard assigner from trainWithKMeans, then creates the annotator and
        the extractor for pulling featurevectors from the images.
        The annotator is then used to train the data
     */
    private static void performTask(GroupedDataset<String, ListDataset<FImage>, FImage> train,
                                VFSListDataset<FImage> testing, GroupedDataset<String, ListDataset<FImage>, FImage> val){

        PyramidDenseSIFT<FImage> sift = new PyramidDenseSIFT<>(
                new DenseSIFT(5, 7), 6f, 7);

        //Build the Vocab and save it in the HardAssigner
        HardAssigner<byte[], float[], IntFloatPair> assigner =
                trainWithKMeans(GroupedUniformRandomisedSampler.sample(train, 30), sift);

        FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(sift, assigner);

        LiblinearAnnotator<FImage, String> annot = svmExtractor(extractor);

        System.out.println("Started training");

        //Training the data
        annot.train(train);

        System.out.println("Finished Training");

        validateVerifier(annot, val);

        //write to file
        File output = new File("run3.txt");
        try {
            if(!output.exists())
                output.createNewFile();
        }catch (Exception e){}

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

            //Classifies the Image and returns a result
            ClassificationResult<String> res = annot.classify(testing.get(i));

            String app = "";
            for(String s: res.getPredictedClasses())
                app += s;

            //Prints out to file and System
            String out = img.getName().getBaseName() + " " + app;
            printer.println(out);

        }
        System.out.println("complete");

    }

    /*
        Implementation of the NaiveBayes annotator, now not used.
     */
    private static NaiveBayesAnnotator<FImage, String> bayesExtractor(FeatureExtractor<DoubleFV, FImage> extr){
        return new NaiveBayesAnnotator<>(extr, NaiveBayesAnnotator.Mode.ALL);
    }

    /*
        Implementation of the HomogenousKernelMap with LibLinearExtractor, now used.
     */
    private static LiblinearAnnotator<FImage, String> svmExtractor(FeatureExtractor<DoubleFV, FImage> extr){
        HomogeneousKernelMap map =
                new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.JensonShannon,
                        HomogeneousKernelMap.WindowType.Rectangular);

        return new LiblinearAnnotator<>(
                map.createWrappedExtractor(extr), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 15.0, 0.00001d);
    }


    /*
        Validation method used throughout the runs. Placed in here to demonstrate how we verified the data.
     */
    private static void validateVerifier(Annotator<FImage, String> annotator,
                                         GroupedDataset<String, ListDataset<FImage>, FImage> validation){
        int corr = 0, size = 0;
        for(Map.Entry<String, ListDataset<FImage>> en: validation.entrySet()){
            for(FImage img: en.getValue()){
                ClassificationResult<String> res = annotator.classify(img);

                String app = "";
                for(String s: res.getPredictedClasses())
                    app += s;

                if(app.equals(en.getKey()))
                    corr++;

//                String out = en.getKey() + ":" + app + ":";
//                System.out.println(out);
            }
            size += en.getValue().size();
        }
        System.out.println("accuracy: " + corr +" "+size);
    }
}
