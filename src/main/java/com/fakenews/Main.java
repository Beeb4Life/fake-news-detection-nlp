package com.fakenews;

import com.fakenews.data.CsvDatasetLoader;
import com.fakenews.data.NewsArticle;
import com.fakenews.feature.FeatureVector;
import com.fakenews.feature.TfidfVectorizer;
import com.fakenews.ml.*;
import com.fakenews.nlp.TextPreprocessor;
import com.fakenews.util.ModelPersistence;
import com.fakenews.util.TrainedPipeline;

import java.io.File;
import java.util.*;

/**
 * Interactive Command-Line Interface for Fake News Detection using NLP.
 * Faithful implementation of ICIDCA 2023 IEEE Research Paper:
 * "Fake News Detection using NLP" (Shaik et al.)
 */
public class Main {

    private static final String DEFAULT_DATASET = "data/sample_news.csv";
    private static final String DEFAULT_MODEL_PATH = "models/trained_model.bin";

    private static TrainedPipeline currentPipeline = null;
    private static EvaluationResult lastEvaluation = null;

    public static void main(String[] args) {
        System.out.println("================================================================");
        System.out.println("     FAKE NEWS DETECTION NLP SYSTEM (ICIDCA 2023 IEEE)          ");
        System.out.println("================================================================");

        // Attempt to auto-load existing saved model if present
        File modelFile = new File(DEFAULT_MODEL_PATH);
        if (modelFile.exists()) {
            try {
                currentPipeline = ModelPersistence.load(modelFile);
                System.out.printf("[INFO] Automatically loaded pre-trained model: %s\n",
                        currentPipeline.getClassifier().getModelName());
                System.out.printf("       Trained Test Accuracy: %.2f%% | Vocabulary: %,d terms\n",
                        currentPipeline.getTestAccuracy() * 100.0,
                        currentPipeline.getVectorizer().getVocabularySize());
            } catch (Exception e) {
                System.out.println("[INFO] No cached model loaded: " + e.getMessage());
            }
        }

        Scanner scanner = new Scanner(System.in);
        boolean running = true;

        while (running) {
            printMenu();
            System.out.print("Enter choice [1-5]: ");
            String choice = scanner.nextLine().trim();

            switch (choice) {
                case "1":
                    handleTrainModel(scanner);
                    break;
                case "2":
                    handleEvaluateModel();
                    break;
                case "3":
                    handlePredictManual(scanner);
                    break;
                case "4":
                    handleShowPaperComparison();
                    break;
                case "5":
                    System.out.println("\nExiting Fake News Detection System. Goodbye!");
                    running = false;
                    break;
                default:
                    System.out.println("\n[!] Invalid selection. Please enter a number between 1 and 5.\n");
            }
        }
    }

    private static void printMenu() {
        System.out.println("\n----------------------------------------------------------------");
        System.out.println(" MAIN MENU");
        System.out.println("----------------------------------------------------------------");
        System.out.printf(" 1. Train Model %s\n",
                (currentPipeline != null ? "(Currently: " + currentPipeline.getClassifier().getModelName() + ")" : "[No Model Trained Yet]"));
        System.out.println(" 2. Evaluate Model (Accuracy, Precision, Recall, F1, Confusion Matrix)");
        System.out.println(" 3. Enter News Article/Headline Manually (Predict REAL / FAKE)");
        System.out.println(" 4. View Research Paper Accuracies & Comparison");
        System.out.println(" 5. Exit");
        System.out.println("----------------------------------------------------------------");
    }

    private static void handleTrainModel(Scanner scanner) {
        System.out.println("\n=== [1] TRAIN MACHINE LEARNING MODEL ===");
        System.out.print("Enter path to dataset CSV [Press Enter for default: " + DEFAULT_DATASET + "]: ");
        String pathInput = scanner.nextLine().trim();
        String datasetPath = pathInput.isEmpty() ? DEFAULT_DATASET : pathInput;

        File dataFile = new File(datasetPath);
        if (!dataFile.exists()) {
            System.out.println("[ERROR] Dataset file not found: " + dataFile.getAbsolutePath());
            System.out.println("Tip: Place your CSV in the 'data/' folder or use the default sample dataset.");
            return;
        }

        System.out.println("\nChoose Classifier Algorithm:");
        System.out.println("  1. Passive-Aggressive Classifier (Paper Top: 97.86% Accuracy) [Recommended]");
        System.out.println("  2. Logistic Regression Classifier (Paper: 96.65% Accuracy)");
        System.out.print("Select algorithm [1 or 2, default=1]: ");
        String algoChoice = scanner.nextLine().trim();

        Classifier classifier;
        if ("2".equals(algoChoice)) {
            classifier = new LogisticRegressionClassifier();
        } else {
            classifier = new PassiveAggressiveClassifier();
        }

        try {
            System.out.println("\n[1/5] Loading and parsing dataset from: " + dataFile.getName() + " ...");
            List<NewsArticle> articles = CsvDatasetLoader.loadFromCsv(dataFile);
            System.out.printf("      Loaded %,d total articles.\n", articles.size());

            long realCount = articles.stream().filter(NewsArticle::isReal).count();
            long fakeCount = articles.stream().filter(NewsArticle::isFake).count();
            System.out.printf("      Distribution: %,d REAL (1) | %,d FAKE (0)\n", realCount, fakeCount);

            if (articles.size() < 10) {
                System.out.println("[ERROR] Dataset has fewer than 10 samples. Training requires at least 10 samples.");
                return;
            }

            // Shuffle for uniform distribution
            Collections.shuffle(articles, new Random(42L));

            // 80% Train, 20% Test split
            int splitIdx = (int) (articles.size() * 0.80);
            List<NewsArticle> trainArticles = articles.subList(0, splitIdx);
            List<NewsArticle> testArticles = articles.subList(splitIdx, articles.size());
            System.out.printf("[2/5] Split dataset: %,d training samples (80%%) | %,d test samples (20%%)\n",
                    trainArticles.size(), testArticles.size());

            // NLP Preprocessing
            System.out.println("[3/5] Preprocessing text (Lowercasing, Cleaning, Stop-words, Porter Stemming) ...");
            TextPreprocessor preprocessor = new TextPreprocessor(true, true);

            List<List<String>> trainTokens = new ArrayList<>(trainArticles.size());
            for (NewsArticle art : trainArticles) {
                trainTokens.add(preprocessor.process(art.getCombinedContent()));
            }

            List<List<String>> testTokens = new ArrayList<>(testArticles.size());
            for (NewsArticle art : testArticles) {
                testTokens.add(preprocessor.process(art.getCombinedContent()));
            }

            // TF-IDF Feature Extraction
            System.out.println("[4/5] Extracting TF-IDF feature representations ...");
            TfidfVectorizer vectorizer = new TfidfVectorizer(10000, 1);
            vectorizer.fit(trainTokens);
            System.out.printf("      Built TF-IDF Vocabulary of %,d unique stemmed terms.\n", vectorizer.getVocabularySize());

            List<FeatureVector> trainFeatures = new ArrayList<>(trainArticles.size());
            int[] trainLabels = new int[trainArticles.size()];
            for (int i = 0; i < trainArticles.size(); i++) {
                trainFeatures.add(vectorizer.transform(trainTokens.get(i)));
                trainLabels[i] = trainArticles.get(i).getLabel();
            }

            List<FeatureVector> testFeatures = new ArrayList<>(testArticles.size());
            int[] testLabels = new int[testArticles.size()];
            for (int i = 0; i < testArticles.size(); i++) {
                testFeatures.add(vectorizer.transform(testTokens.get(i)));
                testLabels[i] = testArticles.get(i).getLabel();
            }

            // Model Training
            System.out.println("[5/5] Training " + classifier.getModelName() + " ...");
            long startTime = System.currentTimeMillis();
            classifier.train(trainFeatures, trainLabels);
            long elapsed = System.currentTimeMillis() - startTime;
            System.out.printf("      Training completed in %d ms.\n", elapsed);

            // Evaluation
            EvaluationResult trainEval = ModelEvaluator.evaluate(classifier, trainFeatures, trainLabels);
            EvaluationResult testEval = ModelEvaluator.evaluate(classifier, testFeatures, testLabels);

            lastEvaluation = testEval;
            currentPipeline = new TrainedPipeline(
                    preprocessor, vectorizer, classifier,
                    trainEval.getAccuracy(), testEval.getAccuracy()
            );

            System.out.println("\n>>> TRAINING FINISHED SUCCESSFULLY <<<");
            System.out.printf("Training Accuracy : %.2f%%\n", trainEval.getAccuracy() * 100.0);
            System.out.printf("Test Accuracy     : %.2f%%\n", testEval.getAccuracy() * 100.0);
            System.out.println(testEval.getFormattedReport());

            // Save Model
            File saveFile = new File(DEFAULT_MODEL_PATH);
            ModelPersistence.save(currentPipeline, saveFile);
            System.out.println("[SAVED] Model and TF-IDF pipeline saved to: " + saveFile.getPath());

        } catch (Exception e) {
            System.out.println("[ERROR] Failed to train model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void handleEvaluateModel() {
        System.out.println("\n=== [2] EVALUATE MODEL ===");
        if (currentPipeline == null) {
            System.out.println("[!] No trained model available. Please train a model (Option 1) first.");
            return;
        }

        if (lastEvaluation != null) {
            System.out.println(lastEvaluation.getFormattedReport());
        } else {
            System.out.println("Current Model: " + currentPipeline.getClassifier().getModelName());
            System.out.printf("Last Recorded Test Accuracy : %.2f%%\n", currentPipeline.getTestAccuracy() * 100.0);
            System.out.printf("Vocabulary Size             : %,d terms\n", currentPipeline.getVectorizer().getVocabularySize());
            System.out.printf("Model trained at            : %s\n", new Date(currentPipeline.getTrainedTimestamp()));
        }
    }

    private static void handlePredictManual(Scanner scanner) {
        System.out.println("\n=== [3] ENTER NEWS ARTICLE / HEADLINE MANUALLY ===");
        if (currentPipeline == null) {
            System.out.println("[!] No trained model available. Please train a model (Option 1) first.");
            return;
        }

        System.out.print("Enter News Headline/Title: ");
        String title = scanner.nextLine().trim();

        System.out.print("Enter News Body Content: ");
        String body = scanner.nextLine().trim();

        if (title.isEmpty() && body.isEmpty()) {
            System.out.println("[!] Headline and Body cannot both be empty.");
            return;
        }

        TrainedPipeline.PredictionResult result = currentPipeline.predict(title, body);

        System.out.println("\n================================================================");
        System.out.println("                      PREDICTION RESULT                         ");
        System.out.println("================================================================");
        if (result.isReal()) {
            System.out.println("  VERDICT       : [REAL NEWS] (1)");
            System.out.printf("  CONFIDENCE    : %.2f%%\n", result.confidence * 100.0);
            System.out.printf("  PROBABILITY   : %.2f%% Real | %.2f%% Fake\n",
                    result.probReal * 100.0, (1.0 - result.probReal) * 100.0);
        } else {
            System.out.println("  VERDICT       : [FAKE NEWS] (0)");
            System.out.printf("  CONFIDENCE    : %.2f%%\n", result.confidence * 100.0);
            System.out.printf("  PROBABILITY   : %.2f%% Fake | %.2f%% Real\n",
                    (1.0 - result.probReal) * 100.0, result.probReal * 100.0);
        }
        System.out.printf("  TOKENS PARSED : %d stemmed keywords extracted\n", result.tokenCount);

        if (!result.topTerms.isEmpty()) {
            System.out.println("\n  TOP CONTRIBUTING NLP INDICATOR TERMS:");
            int count = Math.min(5, result.topTerms.size());
            for (int i = 0; i < count; i++) {
                TrainedPipeline.TermContribution tc = result.topTerms.get(i);
                String direction = tc.score > 0 ? "-> Steers towards REAL" : "-> Steers towards FAKE";
                System.out.printf("    - %-16s (weight contribution: %+.4f) %s\n",
                        "\"" + tc.term + "\"", tc.score, direction);
            }
        }
        System.out.println("================================================================");
    }

    private static void handleShowPaperComparison() {
        System.out.println("\n================================================================");
        System.out.println("    RESEARCH PAPER ACCURACY BENCHMARKS (Shaik et al., 2023)     ");
        System.out.println("================================================================");
        System.out.println(" Algorithm                       Paper Accuracy   Current Model ");
        System.out.println("----------------------------------------------------------------");
        String currentPA = "-";
        String currentLR = "-";

        if (currentPipeline != null) {
            String val = String.format("%.2f%%", currentPipeline.getTestAccuracy() * 100.0);
            if (currentPipeline.getClassifier() instanceof PassiveAggressiveClassifier) {
                currentPA = val + " (Active)";
            } else if (currentPipeline.getClassifier() instanceof LogisticRegressionClassifier) {
                currentLR = val + " (Active)";
            }
        }

        System.out.printf(" 1. Passive-Aggressive (PA)      97.86%% (Best)    %s\n", currentPA);
        System.out.printf(" 2. Logistic Regression (LR)     96.65%%           %s\n", currentLR);
        System.out.println(" 3. Random Forest (RF)           95.81%           -");
        System.out.println(" 4. Decision Tree (DT)           95.29%           -");
        System.out.println("================================================================");
        System.out.println("Key Paper Finding:");
        System.out.println("  The Passive-Aggressive classifier achieved the top performance");
        System.out.println("  (97.86%) due to its fast margin-based aggressive error updates");
        System.out.println("  and high efficiency on sparse TF-IDF text vectors.");
        System.out.println("================================================================");
    }
}
