package com.fakenews.data;

import java.io.Serializable;

/**
 * Represents a single news article with title, content body, and authenticity label.
 * Maps directly to Section III-A and Section III-D of the research paper:
 * - Labels: 0 = FAKE, 1 = REAL
 * - Article representation: Consolidated Title and Body text.
 */
public class NewsArticle implements Serializable {
    private static final long serialVersionUID = 1L;

    public static final int LABEL_FAKE = 0;
    public static final int LABEL_REAL = 1;
    public static final int LABEL_UNKNOWN = -1;

    private final String id;
    private final String title;
    private final String text;
    private final int label; // 0 = FAKE, 1 = REAL

    public NewsArticle(String id, String title, String text, int label) {
        this.id = id != null ? id : "";
        this.title = title != null ? title.trim() : "";
        this.text = text != null ? text.trim() : "";
        this.label = label;
    }

    public NewsArticle(String title, String text) {
        this("", title, text, LABEL_UNKNOWN);
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public String getText() {
        return text;
    }

    public int getLabel() {
        return label;
    }

    public boolean isFake() {
        return label == LABEL_FAKE;
    }

    public boolean isReal() {
        return label == LABEL_REAL;
    }

    /**
     * Consolidates title and body text into a single unified article content string,
     * as described in Section III-D of the research paper.
     */
    public String getCombinedContent() {
        if (title.isEmpty()) {
            return text;
        }
        if (text.isEmpty()) {
            return title;
        }
        return title + " " + text;
    }

    @Override
    public String toString() {
        return String.format("NewsArticle[title='%s', label=%s]",
                title.length() > 50 ? title.substring(0, 47) + "..." : title,
                label == LABEL_REAL ? "REAL" : (label == LABEL_FAKE ? "FAKE" : "UNKNOWN"));
    }
}
