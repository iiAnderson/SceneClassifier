package uk.ac.soton.ecs.ra12ca14.run2;

import de.bwaldvogel.liblinear.*;
import org.apache.commons.vfs2.*;
import org.apache.log4j.*;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.*;
import org.openimaj.experiment.dataset.split.*;
import org.openimaj.experiment.evaluation.classification.*;
import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.image.pixel.sampling.*;
import org.openimaj.math.geometry.shape.*;
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.linear.*;
import org.openimaj.ml.clustering.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.ml.clustering.kmeans.*;
import org.openimaj.util.pair.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * Linear Classifier that uses patches to create featurevectors, which are then used to create a vocab
 * and classify the images.
 */
public class App {

    /*
        Imports and splits the data, which is then used to create the vocab with trainWithKMeans.
        A HardAssinger containing the vocab is returned and used in by the Linear Annotator.
        the annotator is then used to train and classify the data.
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

        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(
                training, 50, 40, 10);


        HardAssigner<float[], float[], IntFloatPair> assigner =
                trainWithKMeans(GroupedUniformRandomisedSampler.sample(training, 30));

        System.out.println("Built Extractor");
        RectangleSamplerExtractor extractor = new RectangleSamplerExtractor(assigner);
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<>(extractor,
                //MultiLabel to make onevsall binary classifier
                LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 15.0, 0.1d);


        annotator.train(splitter.getTrainingDataset());
        System.out.println("Training completed");

        validateVerifier(annotator, splitter.getValidationDataset());

        //Writes ouptut of testing to file
        File output = new File("run2.txt");
        try {
            if(!output.exists())
                output.createNewFile();
        }catch (Exception e){}

        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter(output);
        } catch (IOException e) {
            e.printStackTrace();
        }
        PrintWriter printer = new PrintWriter(fileWriter);

        for(int i = 0; i < testing.size(); i ++){
            FileObject img = testing.getFileObject(i);

            ClassificationResult<String> res = annotator.classify(testing.get(i));

            String app = "";
            for(String s: res.getPredictedClasses())
                app += s;

            String out = "Image " + img.getName().getBaseName() + " predicted as: " + app;
            System.out.println(out);
            printer.println(out);

        }
    }

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

    private static HardAssigner<float[], float[], IntFloatPair> trainWithKMeans(
            Dataset<FImage> sample) {

        List<FloatFV> vec = new ArrayList<>();

        for (FImage image : sample) {
            float tot = 0;
            int pixelTot = 0;
            for(int i = 0; i < image.getWidth(); i++){
                for(int j = 0; j < image.getHeight(); j++){
                    tot += image.pixels[j][i];
                    pixelTot++;
                }
            }

            RectangleSampler sampler = new RectangleSampler(image.subtractInplace(tot/pixelTot).normalise(), 4, 4, 8, 8);

            Iterator<Rectangle> iterator = sampler.iterator();

            while(iterator.hasNext()) {

                Rectangle rec = iterator.next();

                vec.add(new FloatFV(image.extractROI(rec).getFloatPixelVector()));
            }
        }


        float[][] vectors = new float[200000][];

        for(int i = 0; i < 200000; i++){
            vectors[i] = vec.get(i).getVector();
        }

        System.out.println(vectors.length);

        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);

        FloatCentroidsResult result = km.cluster(vectors);

        return result.defaultHardAssigner();
    }
}
