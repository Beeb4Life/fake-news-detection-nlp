package com.fakenews.ml;

import java.io.Serializable;

/**
 * Stores and formats comprehensive evaluation metrics matching Section I and Section IV of the paper:
 * - Accuracy
 * - Precision
 * - Recall
 * - F1 Score
 * - Confusion Matrix
 */
public class EvaluationResult implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String modelName;
    private final int totalSamples;
    private final int truePositives;   // Real predicted as Real
    private final int trueNegatives;   // Fake predicted as Fake
    private final int falsePositives;  // Fake predicted as Real
    private final int falseNegatives;  // Real predicted as Fake

    private final double accuracy;
    private final double precision;
    private final double recall;
    private final double f1Score;

    public EvaluationResult(String modelName, int tp, int tn, int fp, int fn) {
        this.modelName = modelName;
        this.truePositives = tp;
        this.trueNegatives = tn;
        this.falsePositives = fp;
        this.falseNegatives = fn;
        this.totalSamples = tp + tn + fp + fn;

        this.accuracy = totalSamples > 0 ? (double) (tp + tn) / totalSamples : 0.0;
        this.precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        this.recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        this.f1Score = (precision + recall) > 0 ? 2.0 * (precision * recall) / (precision + recall) : 0.0;
    }

    public String getModelName() { return modelName; }
    public int getTotalSamples() { return totalSamples; }
    public int getTruePositives() { return truePositives; }
    public int getTrueNegatives() { return trueNegatives; }
    public int getFalsePositives() { return falsePositives; }
    public int getFalseNegatives() { return falseNegatives; }

    public double getAccuracy() { return accuracy; }
    public double getPrecision() { return precision; }
    public double getRecall() { return recall; }
    public double getF1Score() { return f1Score; }

    /**
     * Formats evaluation report with clean ASCII tables for presentation during project viva.
     */
    public String getFormattedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n============================================================\n");
        sb.append(String.format("   EVALUATION REPORT: %s\n", modelName));
        sb.append("============================================================\n");
        sb.append(String.format(" Total Test Samples : %,d\n", totalSamples));
        sb.append(String.format(" Accuracy           : %.2f%%\n", accuracy * 100.0));
        sb.append(String.format(" Precision (REAL)   : %.2f%%\n", precision * 100.0));
        sb.append(String.format(" Recall (REAL)      : %.2f%%\n", recall * 100.0));
        sb.append(String.format(" F1-Score (REAL)    : %.2f%%\n", f1Score * 100.0));
        sb.append("------------------------------------------------------------\n");
        sb.append(" CONFUSION MATRIX:\n");
        sb.append("                      Predicted FAKE      Predicted REAL\n");
        sb.append(String.format("  Actual FAKE (0)   :      %-16s    %-16s\n",
                String.format("%,d (TN)", trueNegatives), String.format("%,d (FP)", falsePositives)));
        sb.append(String.format("  Actual REAL (1)   :      %-16s    %-16s\n",
                String.format("%,d (FN)", falseNegatives), String.format("%,d (TP)", truePositives)));
        sb.append("============================================================\n");
        return sb.toString();
    }
}
