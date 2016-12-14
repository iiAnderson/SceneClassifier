package uk.ac.soton.ecs.ra12ca14.run3;

import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.image.annotation.evaluation.datasets.*;
import org.openimaj.image.feature.dense.gradient.dsift.*;
import org.openimaj.image.feature.local.aggregate.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.util.pair.*;

/*
    Pyramid Histogram of Words
 */
public class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {

    //Used to pull keypoints from an image
    PyramidDenseSIFT<FImage> siftAnalyser;
    //Contains the vocab
    HardAssigner<byte[], float[], IntFloatPair> assigner;

    public PHOWExtractor(PyramidDenseSIFT<FImage> siftAnalyser, HardAssigner<byte[], float[], IntFloatPair> assigner)
    {
        this.siftAnalyser = siftAnalyser;
        this.assigner = assigner;
    }

    /*
        Performs the PHOW extraction by extracting dense sifts using the PyramidDenseSIFT by analysing the images
        and later extracting the keypoints from the image using byteKeypoints.
        The Bag Of Visual Words takes the HardAssigner, which contains the vocabulary built using KNN,
        The Aggregator uses the BOVW vocabulary to build a pyramid of the features, and then aggregates this
        into a single Sparse Vector, and is then normalised into a DoubleFV.
     */
    public DoubleFV extractFeature(FImage object) {
        FImage image = object.getImage();

        siftAnalyser.analyseImage(image);

        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

        PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<>(
                bovw, 2, 4, 6);

        return spatial.aggregate(siftAnalyser.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
    }
}