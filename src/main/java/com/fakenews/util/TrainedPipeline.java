package com.fakenews.util;

import com.fakenews.feature.FeatureVector;
import com.fakenews.feature.TfidfVectorizer;
import com.fakenews.ml.Classifier;
import com.fakenews.nlp.TextPreprocessor;

import java.io.Serializable;
import java.util.*;

/**
 * Encapsulates the entire end-to-end trained inference pipeline:
 *  1. TextPreprocessor
 *  2. TfidfVectorizer
 *  3. Classifier
 */
public class TrainedPipeline implements Serializable {
    private static final long serialVersionUID = 1L;

    private final TextPreprocessor preprocessor;
    private final TfidfVectorizer vectorizer;
    private final Classifier classifier;
    private final double trainingAccuracy;
    private final double testAccuracy;
    private final long trainedTimestamp;

    public TrainedPipeline(TextPreprocessor preprocessor,
                           TfidfVectorizer vectorizer,
                           Classifier classifier,
                           double trainingAccuracy,
                           double testAccuracy) {
        this.preprocessor = preprocessor;
        this.vectorizer = vectorizer;
        this.classifier = classifier;
        this.trainingAccuracy = trainingAccuracy;
        this.testAccuracy = testAccuracy;
        this.trainedTimestamp = System.currentTimeMillis();
    }

    public TextPreprocessor getPreprocessor() { return preprocessor; }
    public TfidfVectorizer getVectorizer() { return vectorizer; }
    public Classifier getClassifier() { return classifier; }
    public double getTrainingAccuracy() { return trainingAccuracy; }
    public double getTestAccuracy() { return testAccuracy; }
    public long getTrainedTimestamp() { return trainedTimestamp; }

    /**
     * Runs inference on raw text (title + body) and returns a detailed prediction.
     */
    public PredictionResult predict(String title, String body) {
        String combined = (title != null ? title : "") + " " + (body != null ? body : "");
        List<String> tokens = preprocessor.process(combined);
        FeatureVector feature = vectorizer.transform(tokens);

        int predLabel = classifier.predict(feature);
        double probReal = classifier.predictProbability(feature);
        double confidence = predLabel == 1 ? probReal : (1.0 - probReal);

        // Find top contributing terms in the input article
        List<TermContribution> contributions = new ArrayList<>();
        double[] weights = classifier.getWeights();
        int[] indices = feature.getIndices();
        double[] values = feature.getValues();

        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            double weight = idx < weights.length ? weights[idx] : 0.0;
            double score = values[i] * weight;
            String term = vectorizer.getTerm(idx);
            if (term != null) {
                contributions.add(new TermContribution(term, score));
            }
        }

        // Sort by magnitude of contribution
        contributions.sort((a, b) -> Double.compare(Math.abs(b.score), Math.abs(a.score)));

        return new PredictionResult(predLabel, probReal, confidence, tokens.size(), contributions);
    }

    public static class PredictionResult {
        public final int label; // 0 = FAKE, 1 = REAL
        public final double probReal;
        public final double confidence;
        public final int tokenCount;
        public final List<TermContribution> topTerms;

        public PredictionResult(int label, double probReal, double confidence, int tokenCount, List<TermContribution> topTerms) {
            this.label = label;
            this.probReal = probReal;
            this.confidence = confidence;
            this.tokenCount = tokenCount;
            this.topTerms = topTerms;
        }

        public boolean isReal() { return label == 1; }
        public boolean isFake() { return label == 0; }
        public String getLabelString() { return label == 1 ? "REAL" : "FAKE"; }
    }

    public static class TermContribution implements Serializable {
        private static final long serialVersionUID = 1L;
        public final String term;
        public final double score;

        public TermContribution(String term, double score) {
            this.term = term;
            this.score = score;
        }
    }
}
