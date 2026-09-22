package com.fakenews.feature;

import java.io.Serializable;
import java.util.Map;

/**
 * High-performance sparse vector representation for document TF-IDF features.
 * Stores non-zero feature indices and their corresponding weights.
 */
public class FeatureVector implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int[] indices;
    private final double[] values;

    public FeatureVector(int[] indices, double[] values) {
        this.indices = indices;
        this.values = values;
    }

    public FeatureVector(Map<Integer, Double> featureMap) {
        this.indices = new int[featureMap.size()];
        this.values = new double[featureMap.size()];
        int i = 0;
        for (Map.Entry<Integer, Double> entry : featureMap.entrySet()) {
            indices[i] = entry.getKey();
            values[i] = entry.getValue();
            i++;
        }
    }

    public int[] getIndices() {
        return indices;
    }

    public double[] getValues() {
        return values;
    }

    public int size() {
        return indices.length;
    }

    /**
     * Computes the dot product between this sparse vector and a dense weight vector w.
     */
    public double dot(double[] weights) {
        double sum = 0.0;
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (idx >= 0 && idx < weights.length) {
                sum += values[i] * weights[idx];
            }
        }
        return sum;
    }

    /**
     * Computes the squared L2 norm ||x||^2 of this sparse vector.
     */
    public double squaredNorm() {
        double sum = 0.0;
        for (double val : values) {
            sum += val * val;
        }
        return sum;
    }
}
