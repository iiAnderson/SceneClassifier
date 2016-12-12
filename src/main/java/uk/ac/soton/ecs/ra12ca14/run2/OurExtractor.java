package uk.ac.soton.ecs.ra12ca14.run2;

import org.openimaj.feature.*;
import org.openimaj.image.*;
import org.openimaj.image.feature.local.aggregate.*;
import org.openimaj.image.pixel.sampling.*;
import org.openimaj.knn.*;
import org.openimaj.knn.approximate.*;
import org.openimaj.math.geometry.shape.*;
import org.openimaj.ml.clustering.assignment.*;
import org.openimaj.ml.clustering.kmeans.*;
import org.openimaj.util.pair.*;

import java.util.*;

public class OurExtractor implements FeatureExtractor<DoubleFV, FImage> {

    HardAssigner<byte[], float[], IntFloatPair> assigner;

    public OurExtractor(HardAssigner<byte[], float[], IntFloatPair> assigner) {
        this.assigner = assigner;
    }

    public DoubleFV extractFeature(FImage image) {

        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

        RectangleSampler sampler = new RectangleSampler(image, 4, 4, 8, 8);

        Iterator<Rectangle> iterator = sampler.iterator();
        List<float[]> list = new ArrayList<>();

        while(iterator.hasNext()){
            Rectangle rec = iterator.next();
            float[][] pixels = new float[(int) rec.getWidth()][(int) rec.getHeight()];
            for(int x = (int) rec.minX(); x < rec.getWidth(); x++){
                for(int y = (int) rec.minY(); y < rec.getHeight(); y++){
                    pixels[x][y] = image.pixels[x][y];
                }
            }
            list.add(new FImage(pixels).getFloatPixelVector());
        }

        float[][] data = new float[list.size()][64];
        data = list.toArray(data);


        KMeansConfiguration<FloatNearestNeighboursKDTree, float[]> config =
                new KMeansConfiguration<>(500, new FloatNearestNeighboursKDTree.Factory());


        FloatKMeans kmeans = new FloatKMeans(config);
        FeatureVector[] toAgg = kmeans.cluster(list).getCentroids();
        DoubleFV[] agg = new DoubleFV[toAgg.length];

        for(int i = 0; i < toAgg.length; i++) {
            agg[i] = toAgg[i].asDoubleFV();
        }


        return bovw.aggregate(Arrays.asList(agg));
    }
}