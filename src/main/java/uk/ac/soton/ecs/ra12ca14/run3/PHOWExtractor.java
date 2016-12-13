package uk.ac.soton.ecs.ra12ca14.run3;

import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.image.annotation.evaluation.datasets.*;
import org.openimaj.image.feature.dense.gradient.dsift.*;
import org.openimaj.image.feature.local.aggregate.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.util.pair.*;

/**
 * OpenIMAJ Tutorial 12 - Chloe Allan
 * PHOW Extractor implementation taken from the tutorial to train our classifier. It uses a BlockSpatialAggregator
 * with a BagOfVisualWords to get 4 histograms for the image. The histograms are appended and then normalised, and
 * then returned.
 */
public class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {

    PyramidDenseSIFT<FImage> pdsift;
    HardAssigner<byte[], float[], IntFloatPair> assigner;

    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
    {
        this.pdsift = pdsift;
        this.assigner = assigner;
    }

    public DoubleFV extractFeature(FImage object) {
        FImage image = object.getImage();
        pdsift.analyseImage(image);

        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);


        PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<>(
                bovw, 2, 4);

        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
    }
}