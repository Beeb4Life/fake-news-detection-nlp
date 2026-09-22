package com.fakenews.ml;

import com.fakenews.feature.FeatureVector;

import java.io.Serializable;
import java.util.List;

/**
 * Common interface for Fake News Machine Learning Classifiers.
 */
public interface Classifier extends Serializable {

    /**
     * Trains the classifier on the provided feature vectors and binary labels (0 = FAKE, 1 = REAL).
     */
    void train(List<FeatureVector> features, int[] labels);

    /**
     * Predicts binary class label: 0 for FAKE, 1 for REAL.
     */
    int predict(FeatureVector feature);

    /**
     * Computes raw decision score or calibrated probability of being REAL (1).
     * Score > 0.0 indicates REAL, score <= 0.0 indicates FAKE.
     */
    double decisionFunction(FeatureVector feature);

    /**
     * Returns the estimated probability (between 0.0 and 1.0) of being REAL.
     */
    double predictProbability(FeatureVector feature);

    /**
     * Name of the algorithm (e.g., "Passive-Aggressive", "Logistic Regression").
     */
    String getModelName();

    /**
     * Feature weights array.
     */
    double[] getWeights();

    /**
     * Intercept / bias term.
     */
    double getBias();
}
