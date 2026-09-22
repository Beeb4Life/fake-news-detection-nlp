package com.fakenews.ml;

import com.fakenews.feature.FeatureVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Logistic Regression Classifier with Stochastic Gradient Descent (SGD) and L2 Regularization.
 * Evaluated as the second model in the ICIDCA 2023 paper (Shaik et al.), achieving 96.65% accuracy.
 */
public class LogisticRegressionClassifier implements Classifier {
    private static final long serialVersionUID = 1L;

    private final double initialLearningRate;
    private final double l2Penalty;
    private final int maxEpochs;
    private final long randomSeed;

    private double[] weights;
    private double bias;
    private int numFeatures;

    public LogisticRegressionClassifier() {
        this(0.1, 1e-4, 30, 42L);
    }

    public LogisticRegressionClassifier(double learningRate, double l2Penalty, int maxEpochs, long randomSeed) {
        this.initialLearningRate = learningRate;
        this.l2Penalty = l2Penalty;
        this.maxEpochs = maxEpochs;
        this.randomSeed = randomSeed;
    }

    @Override
    public void train(List<FeatureVector> features, int[] labels) {
        if (features.isEmpty()) {
            throw new IllegalArgumentException("Training set is empty");
        }

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

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            Collections.shuffle(indices, rng);
            // Dynamic learning rate decay
            double eta = initialLearningRate / (1.0 + 0.05 * epoch);

            for (int idx : indices) {
                FeatureVector x = features.get(idx);
                double y = labels[idx]; // 0.0 or 1.0

                double z = x.dot(weights) + bias;
                double p = sigmoid(z);
                double error = p - y;

                int[] xIndices = x.getIndices();
                double[] xValues = x.getValues();

                for (int j = 0; j < xIndices.length; j++) {
                    int featureIdx = xIndices[j];
                    double grad = error * xValues[j] + l2Penalty * weights[featureIdx];
                    weights[featureIdx] -= eta * grad;
                }
                bias -= eta * error;
            }
        }
    }

    private double sigmoid(double z) {
        if (z > 20.0) return 1.0;
        if (z < -20.0) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public int predict(FeatureVector feature) {
        return predictProbability(feature) >= 0.5 ? 1 : 0;
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
        return sigmoid(decisionFunction(feature));
    }

    @Override
    public String getModelName() {
        return "Logistic Regression (Paper: 96.65%)";
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
