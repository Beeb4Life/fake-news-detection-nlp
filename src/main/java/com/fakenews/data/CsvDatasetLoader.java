package com.fakenews.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Loads and parses news datasets from CSV files.
 * Supports:
 *  1. 3-column format: (title, text, label) - e.g. Kaggle news.csv (6335 rows)
 *  2. 4-column format: (title, text, subject, date) - e.g. ISOT True.csv / Fake.csv
 *  3. Header-agnostic and case-insensitive column discovery.
 */
public class CsvDatasetLoader {

    /**
     * Loads articles from a single CSV file with a label column.
     */
    public static List<NewsArticle> loadFromCsv(File file) throws IOException {
        if (!file.exists()) {
            throw new IOException("Dataset file not found: " + file.getAbsolutePath());
        }

        List<NewsArticle> articles = new ArrayList<>();
        CSVFormat format = CSVFormat.DEFAULT.builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreHeaderCase(true)
                .setTrim(true)
                .setIgnoreEmptyLines(true)
                .build();

        try (Reader reader = new FileReader(file, StandardCharsets.UTF_8);
             CSVParser parser = format.parse(reader)) {

            for (CSVRecord record : parser) {
                String title = getField(record, "title");
                String text = getField(record, "text", "article", "content", "news");
                String labelStr = getField(record, "label", "target", "class");

                if ((title.isEmpty() && text.isEmpty()) || labelStr.isEmpty()) {
                    continue;
                }

                int label = parseLabel(labelStr);
                if (label == NewsArticle.LABEL_UNKNOWN) {
                    continue;
                }

                String id = getField(record, "id", "unnamed: 0", "index");
                articles.add(new NewsArticle(id, title, text, label));
            }
        }

        return articles;
    }

    /**
     * Loads and merges separate Real and Fake CSV files (e.g., True.csv and Fake.csv from Kaggle ISOT dataset).
     */
    public static List<NewsArticle> loadSeparateFiles(File realFile, File fakeFile) throws IOException {
        List<NewsArticle> articles = new ArrayList<>();
        if (realFile != null && realFile.exists()) {
            articles.addAll(loadFixedLabelFile(realFile, NewsArticle.LABEL_REAL));
        }
        if (fakeFile != null && fakeFile.exists()) {
            articles.addAll(loadFixedLabelFile(fakeFile, NewsArticle.LABEL_FAKE));
        }
        Collections.shuffle(articles);
        return articles;
    }

    private static List<NewsArticle> loadFixedLabelFile(File file, int fixedLabel) throws IOException {
        List<NewsArticle> list = new ArrayList<>();
        CSVFormat format = CSVFormat.DEFAULT.builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreHeaderCase(true)
                .setTrim(true)
                .setIgnoreEmptyLines(true)
                .build();

        try (Reader reader = new FileReader(file, StandardCharsets.UTF_8);
             CSVParser parser = format.parse(reader)) {

            for (CSVRecord record : parser) {
                String title = getField(record, "title");
                String text = getField(record, "text", "article", "content");
                if (title.isEmpty() && text.isEmpty()) {
                    continue;
                }
                list.add(new NewsArticle("", title, text, fixedLabel));
            }
        }
        return list;
    }

    private static String getField(CSVRecord record, String... possibleHeaders) {
        for (String h : possibleHeaders) {
            if (record.isMapped(h)) {
                String val = record.get(h);
                if (val != null) {
                    return val.trim();
                }
            }
        }
        return "";
    }

    /**
     * Converts various label representations to binary: 0 = FAKE, 1 = REAL.
     */
    public static int parseLabel(String raw) {
        if (raw == null) return NewsArticle.LABEL_UNKNOWN;
        String clean = raw.trim().toLowerCase(Locale.ROOT);
        if (clean.equals("1") || clean.equals("real") || clean.equals("true") || clean.equals("reliable")) {
            return NewsArticle.LABEL_REAL;
        }
        if (clean.equals("0") || clean.equals("fake") || clean.equals("false") || clean.equals("unreliable")) {
            return NewsArticle.LABEL_FAKE;
        }
        return NewsArticle.LABEL_UNKNOWN;
    }
}
