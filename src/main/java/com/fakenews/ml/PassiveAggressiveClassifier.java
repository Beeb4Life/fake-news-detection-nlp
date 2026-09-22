package com.fakenews.ml;

import com.fakenews.feature.FeatureVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Passive-Aggressive Classifier (PA-I with Slack Parameter C).
 * Matches the primary algorithm evaluated in the ICIDCA 2023 paper (Shaik et al.),
 * which achieved the highest accuracy of 97.86%.
 *
 * Algorithm Reference:
 *   Crammer, K. et al. (2006). "Online Passive-Aggressive Algorithms", JMLR 7, 551-585.
 *
 * Mechanism:
 *   - Passive: If example is correctly classified with margin >= 1, weights are unchanged (loss = 0).
 *   - Aggressive: If margin < 1, weights are updated aggressively to rectify error with step size tau:
 *       tau = min(C, loss / (||x||^2 + 1))
 *       w = w + tau * y * x
 */
public class PassiveAggressiveClassifier implements Classifier {
    private static final long serialVersionUID = 1L;

    private final double c;          // Aggressiveness / regularization parameter
    private final int maxEpochs;     // Number of training passes
    private final long randomSeed;

    private double[] weights;
    private double bias;
    private int numFeatures;

    public PassiveAggressiveClassifier() {
        this(1.0, 25, 42L);
    }

    public PassiveAggressiveClassifier(double c, int maxEpochs, long randomSeed) {
        this.c = c;
        this.maxEpochs = maxEpochs;
        this.randomSeed = randomSeed;
    }

    @Override
    public void train(List<FeatureVector> features, int[] labels) {
        if (features.isEmpty()) {
            throw new IllegalArgumentException("Training set is empty");
        }

        // Determine feature dimensionality
        int maxDim = 0;
        for (FeatureVector fv : features) {
            for (int idx : fv.getIndices()) {
                if (idx > maxDim) {
                    maxDim = idx;
                }
            }
        }
        this.numFeatures = maxDim + 1;
        this.weights = new double[numFeatures];
        this.bias = 0.0;

        int n = features.size();
        List<Integer> indices = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            indices.add(i);
        }

        Random rng = new Random(randomSeed);

        // Multi-epoch online learning
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            Collections.shuffle(indices, rng);

            for (int idx : indices) {
                FeatureVector x = features.get(idx);
                // Map binary labels {0, 1} to {-1, +1}
                double y = (labels[idx] == 1) ? 1.0 : -1.0;

                double score = x.dot(weights) + bias;
                double margin = y * score;
                double loss = Math.max(0.0, 1.0 - margin);

                if (loss > 0.0) {
                    // Update step size tau (PA-I formulation)
                    double normSq = x.squaredNorm() + 1.0; // +1.0 for bias term
                    double tau = Math.min(c, loss / normSq);

                    // Aggressive weight update
                    double step = tau * y;
                    int[] xIndices = x.getIndices();
                    double[] xValues = x.getValues();
                    for (int j = 0; j < xIndices.length; j++) {
                        weights[xIndices[j]] += step * xValues[j];
                    }
                    bias += step;
                }
            }
        }
    }

    @Override
    public int predict(FeatureVector feature) {
        return decisionFunction(feature) >= 0.0 ? 1 : 0;
    }

    @Override
    public double decisionFunction(FeatureVector feature) {
        if (weights == null) {
            throw new IllegalStateException("Model has not been trained yet");
        }
        return feature.dot(weights) + bias;
    }

    @Override
    public double predictProbability(FeatureVector feature) {
        double score = decisionFunction(feature);
        // Sigmoid probability calibration
        return 1.0 / (1.0 + Math.exp(-score));
    }

    @Override
    public String getModelName() {
        return "Passive-Aggressive Classifier (Paper Top: 97.86%)";
    }

    @Override
    public double[] getWeights() {
        return weights;
    }

    @Override
    public double getBias() {
        return bias;
    }
}
