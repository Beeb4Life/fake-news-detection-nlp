# Fake News Detection using NLP (Java)

A complete, faithful Java implementation of the research paper:
> **"Fake News Detection using NLP"**  
> *Mohammed Ali Shaik, Makkaji Yasha Sree, Sanka Sri Vyshnavi, Thogiti Ganesh, Dasari Sushmitha, Narmetta Shreya*  
> *IEEE International Conference on Innovative Data Communication Technologies and Application (ICIDCA 2023)*  
> DOI: `10.1109/ICIDCA56705.2023.10100305`

---

## Architecture & Methodology

```text
Raw News Article (Title + Body)
               │
               ▼
   [ Text Preprocessing ]
   ├── 1. Case Folding (Lowercase)
   ├── 2. Data Cleaning (HTML, URL, punctuation & digit stripping)
   ├── 3. Tokenization (Whitespace word splitting)
   ├── 4. Stop-Word Removal (NLTK English stop words)
   └── 5. Morphological Normalization (Porter Stemming)
               │
               ▼
   [ Feature Extraction ]
   ├── TF-IDF Vectorization: Smooth IDF = ln((1+N)/(1+df)) + 1
   └── L2 Euclidean Normalization: v = v / ||v||_2
               │
               ▼
   [ Supervised Machine Learning ]
   ├── Passive-Aggressive Classifier (PA-I) -> 97.86% in Paper (Top Model)
   └── Logistic Regression (SGD)            -> 96.65% in Paper
               │
               ▼
   [ Evaluation & Inference ]
   ├── Accuracy, Precision, Recall, F1-Score, Confusion Matrix
   └── Real-time REAL / FAKE Prediction + Contributing Indicator Terms
```

---

## Quick Start

### 1. Requirements
* Java JDK 17 or higher
* Apache Maven (`sudo apt install maven`)

### 2. Build the Project
```bash
./compile.sh
# OR via Maven directly:
mvn clean package -DskipTests
```

### 3. Run the Application
```bash
./run.sh
# OR run the packaged standalone JAR:
java -jar target/fake-news-detector-1.0.0.jar
```
