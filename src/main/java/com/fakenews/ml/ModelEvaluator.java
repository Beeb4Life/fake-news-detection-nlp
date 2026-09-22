package com.fakenews.ml;

import com.fakenews.feature.FeatureVector;

import java.util.List;

/**
 * Evaluates trained classifiers on unseen test data and calculates standard NLP metrics.
 */
public class ModelEvaluator {

    public static EvaluationResult evaluate(Classifier classifier, List<FeatureVector> testFeatures, int[] testLabels) {
        if (testFeatures.size() != testLabels.length) {
            throw new IllegalArgumentException("Feature size and label length mismatch");
        }

        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;

        for (int i = 0; i < testFeatures.size(); i++) {
            int actual = testLabels[i];
            int predicted = classifier.predict(testFeatures.get(i));

            if (actual == 1 && predicted == 1) {
                tp++;
            } else if (actual == 0 && predicted == 0) {
                tn++;
            } else if (actual == 0 && predicted == 1) {
                fp++;
            } else if (actual == 1 && predicted == 0) {
                fn++;
            }
        }

        return new EvaluationResult(classifier.getModelName(), tp, tn, fp, fn);
    }
}
