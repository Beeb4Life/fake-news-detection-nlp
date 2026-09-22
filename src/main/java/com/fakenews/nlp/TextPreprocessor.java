package com.fakenews.nlp;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Implements the NLP text preprocessing pipeline matching Section III-B of the paper:
 * 1. Data Cleaning: HTML and URL removal, special characters and punctuation removal.
 * 2. Case Folding: Lowercasing all text.
 * 3. Text Normalization: Numeral/digit deletion, word stemming via Porter Stemmer.
 * 4. Tokenization: Splitting into word tokens.
 * 5. Stop-words removal: Filtering out frequent English stop words.
 */
public class TextPreprocessor implements Serializable {
    private static final long serialVersionUID = 1L;

    private static final Pattern URL_PATTERN = Pattern.compile("https?://\\S+|www\\.\\S+");
    private static final Pattern HTML_PATTERN = Pattern.compile("<[^>]+>");
    private static final Pattern NON_ALPHA_PATTERN = Pattern.compile("[^a-zA-Z\\s]");
    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");

    private final boolean useStemming;
    private final boolean removeStopWords;

    public TextPreprocessor() {
        this(true, true);
    }

    public TextPreprocessor(boolean useStemming, boolean removeStopWords) {
        this.useStemming = useStemming;
        this.removeStopWords = removeStopWords;
    }

    /**
     * Executes the full NLP preprocessing pipeline on raw news text.
     *
     * @param rawText the combined title and body of the news article
     * @return list of cleaned, stemmed, non-stopword tokens
     */
    public List<String> process(String rawText) {
        if (rawText == null || rawText.trim().isEmpty()) {
            return new ArrayList<>();
        }

        // 1. Lowercase
        String text = rawText.toLowerCase(Locale.ROOT);

        // 2. Remove URLs
        text = URL_PATTERN.matcher(text).replaceAll(" ");

        // 3. Remove HTML tags
        text = HTML_PATTERN.matcher(text).replaceAll(" ");

        // 4. Remove special characters and numerals (keep only alphabets and spaces)
        text = NON_ALPHA_PATTERN.matcher(text).replaceAll(" ");

        // 5. Tokenize on whitespace
        String[] tokens = WHITESPACE_PATTERN.split(text.trim());

        List<String> cleanedTokens = new ArrayList<>(tokens.length);
        for (String token : tokens) {
            String word = token.trim();
            if (word.length() <= 2) {
                continue;
            }

            // 6. Stop-word removal
            if (removeStopWords && StopWords.isStopWord(word)) {
                continue;
            }

            // 7. Morphological normalization via Porter Stemmer
            if (useStemming) {
                word = PorterStemmer.stemWord(word);
            }

            if (word != null && word.length() >= 2) {
                cleanedTokens.add(word);
            }
        }

        return cleanedTokens;
    }
}
