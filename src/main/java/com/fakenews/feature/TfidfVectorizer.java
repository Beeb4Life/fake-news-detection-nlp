package com.fakenews.feature;

import java.io.Serializable;
import java.util.*;

/**
 * TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer.
 * Fits vocabulary and IDF weights on training corpus, transforms tokenized documents
 * into normalized sparse numerical feature vectors.
 *
 * Mathematical formulation (smooth IDF):
 *   idf(t) = ln((1 + N) / (1 + df(t))) + 1.0
 *   tf(t, d) = count(t, d)
 *   weight(t, d) = tf(t, d) * idf(t)
 *   Normalized by Euclidean L2-norm: v = v / ||v||_2
 */
public class TfidfVectorizer implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int maxFeatures;
    private final int minDocFreq;

    private Map<String, Integer> vocabulary;
    private String[] indexToTerm;
    private double[] idfWeights;
    private boolean isFitted = false;

    public TfidfVectorizer() {
        this(5000, 2);
    }

    public TfidfVectorizer(int maxFeatures, int minDocFreq) {
        this.maxFeatures = maxFeatures;
        this.minDocFreq = minDocFreq;
        this.vocabulary = new HashMap<>();
    }

    /**
     * Fits the vocabulary and computes IDF weights across the training corpus.
     */
    public void fit(List<List<String>> tokenizedCorpus) {
        int numDocs = tokenizedCorpus.size();
        if (numDocs == 0) {
            throw new IllegalArgumentException("Cannot fit on empty corpus");
        }

        // 1. Compute document frequency (df) for each unique term
        Map<String, Integer> docFreq = new HashMap<>();
        for (List<String> doc : tokenizedCorpus) {
            Set<String> uniqueTerms = new HashSet<>(doc);
            for (String term : uniqueTerms) {
                docFreq.put(term, docFreq.getOrDefault(term, 0) + 1);
            }
        }

        // 2. Filter terms by minDocFreq and rank by overall frequency
        List<Map.Entry<String, Integer>> filteredTerms = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : docFreq.entrySet()) {
            if (entry.getValue() >= minDocFreq) {
                filteredTerms.add(entry);
            }
        }

        // Sort descending by document frequency
        filteredTerms.sort((a, b) -> Integer.compare(b.getValue(), a.getValue()));

        // Limit to maxFeatures if needed
        int vocabSize = Math.min(filteredTerms.size(), maxFeatures);
        vocabulary = new HashMap<>(vocabSize);
        indexToTerm = new String[vocabSize];
        idfWeights = new double[vocabSize];

        for (int i = 0; i < vocabSize; i++) {
            Map.Entry<String, Integer> entry = filteredTerms.get(i);
            String term = entry.getKey();
            int df = entry.getValue();

            vocabulary.put(term, i);
            indexToTerm[i] = term;

            // Standard smooth IDF formula: ln((1 + N) / (1 + df)) + 1
            idfWeights[i] = Math.log((1.0 + numDocs) / (1.0 + df)) + 1.0;
        }

        isFitted = true;
    }

    /**
     * Transforms a list of tokens into a sparse L2-normalized TF-IDF FeatureVector.
     */
    public FeatureVector transform(List<String> tokens) {
        if (!isFitted) {
            throw new IllegalStateException("Vectorizer has not been fitted yet");
        }

        // 1. Calculate term frequencies (TF) for in-vocabulary words
        Map<Integer, Double> termCounts = new HashMap<>();
        for (String token : tokens) {
            Integer termIdx = vocabulary.get(token);
            if (termIdx != null) {
                termCounts.put(termIdx, termCounts.getOrDefault(termIdx, 0.0) + 1.0);
            }
        }

        if (termCounts.isEmpty()) {
            return new FeatureVector(new int[0], new double[0]);
        }

        // 2. Compute TF-IDF weights: tf * idf
        int numTerms = termCounts.size();
        int[] indices = new int[numTerms];
        double[] values = new double[numTerms];
        int pos = 0;
        double normSquared = 0.0;

        for (Map.Entry<Integer, Double> entry : termCounts.entrySet()) {
            int idx = entry.getKey();
            double tf = entry.getValue();
            double tfidf = tf * idfWeights[idx];

            indices[pos] = idx;
            values[pos] = tfidf;
            normSquared += tfidf * tfidf;
            pos++;
        }

        // 3. L2 Normalization: vector / ||vector||_2
        if (normSquared > 0.0) {
            double l2Norm = Math.sqrt(normSquared);
            for (int i = 0; i < values.length; i++) {
                values[i] /= l2Norm;
            }
        }

        return new FeatureVector(indices, values);
    }

    public int getVocabularySize() {
        return isFitted ? vocabulary.size() : 0;
    }

    public String getTerm(int index) {
        if (indexToTerm != null && index >= 0 && index < indexToTerm.length) {
            return indexToTerm[index];
        }
        return null;
    }

    public Integer getIndex(String term) {
        return vocabulary != null ? vocabulary.get(term) : null;
    }

    public boolean isFitted() {
        return isFitted;
    }
}
