# Fake News Detection using NLP in Java

A Java implementation of the IEEE research paper **"Fake News Detection using NLP"**, presented at the **2023 International Conference on Innovative Data Communication Technologies and Application (ICIDCA)**.

**Paper:** *Fake News Detection using NLP*
**Authors:** Mohammed Ali Shaik et al.
**DOI:** [10.1109/ICIDCA56705.2023.10100305](https://doi.org/10.1109/ICIDCA56705.2023.10100305)

---

## Overview

This project detects whether a news article or headline is **REAL** or **FAKE** using Natural Language Processing and Machine Learning.

The implementation follows the core methodology of the referenced research paper:

* Text cleaning and normalization
* Tokenization and stop-word removal
* Porter stemming
* Smooth TF-IDF feature extraction
* L2 normalization
* Passive-Aggressive (PA-I) classification
* Logistic Regression using SGD
* Model evaluation
* Real-time prediction with contributing keyword indicators

The entire system is implemented in **Java** and runs through a command-line interface.

---

## Application

The application provides an interactive CLI with the following options:

```text
============================================================
        FAKE NEWS DETECTION NLP SYSTEM
============================================================

1. Train Model
2. Evaluate Model
3. Enter News Article / Headline
4. Research Paper Comparison
5. Exit

Enter choice [1-5]:
```

### Main Features

* Train a model using a CSV dataset
* Choose between Passive-Aggressive and Logistic Regression
* Evaluate the trained model
* Enter news headlines/articles for prediction
* Display REAL/FAKE classification and confidence
* Show terms contributing to the prediction
* Save and load trained models

---

## Tech Stack

| Component          | Technology                                       |
| ------------------ | ------------------------------------------------ |
| Language           | Java 17+                                         |
| Build Tool         | Apache Maven                                     |
| Dataset Parsing    | Apache Commons CSV                               |
| NLP                | Custom Java NLP Pipeline                         |
| Stemming           | Porter Stemmer                                   |
| Feature Extraction | Smooth TF-IDF                                    |
| Normalization      | L2 Normalization                                 |
| Classifiers        | Passive-Aggressive PA-I, Logistic Regression SGD |
| Model Persistence  | Java Serialization                               |
| Packaging          | Maven Shade Plugin                               |

---

## Architecture

```text
                News Article / Headline
                         │
                         ▼
              ┌─────────────────────┐
              │  Text Preprocessing │
              │                     │
              │ • Lowercase         │
              │ • Cleaning          │
              │ • Tokenization      │
              │ • Stop-word removal │
              │ • Porter Stemming   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Feature Extraction │
              │                     │
              │ • Smooth TF-IDF     │
              │ • L2 Normalization  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Classification   │
              │                     │
              │ • Passive-Aggressive│
              │ • Logistic Regression│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │      Prediction     │
              │                     │
              │ • REAL / FAKE       │
              │ • Confidence        │
              │ • Key Terms         │
              └─────────────────────┘
```

---

## Methodology

### Smooth TF-IDF

$$
\mathrm{TF}(t,d)=
\frac{f_{t,d}}
{\sum_{t'\in d}f_{t',d}}
$$

$$
\mathrm{IDF}(t)=
\ln\left(\frac{1+N}{1+\mathrm{df}(t)}\right)+1
$$

$$
\mathrm{TF\!-\!IDF}(t,d)=
\mathrm{TF}(t,d)\times\mathrm{IDF}(t)
$$

$$
\mathbf{v}_{norm}=
\frac{\mathbf{v}}{\|\mathbf{v}\|_2}
$$

### Passive-Aggressive (PA-I)

$$
\ell_t=
\max\left(0,1-y_t(\mathbf{w}_t\cdot\mathbf{x}_t)\right)
$$

$$
\tau_t=
\frac{\ell_t}
{\|\mathbf{x}_t\|^2+\frac{1}{2C}}
$$

---

## Working

1. **Preprocessing**
   The input text is cleaned, converted to lowercase, tokenized, and stripped of stop words.

2. **Stemming**
   Porter stemming reduces words to their root forms.

3. **Feature Extraction**
   The processed text is converted into TF-IDF vectors and L2-normalized.

4. **Model Training**
   The selected classifier learns from the training dataset.

5. **Prediction**
   New articles are classified as REAL or FAKE.

6. **Explanation**
   The system displays important terms contributing to the prediction.

7. **Evaluation**
   Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix are calculated.

---

## Installation

### Requirements

* Java JDK 17+
* Apache Maven 3.8+

Check your installation:

```bash
java -version
mvn -version
```

### Clone the Repository

```bash
git clone https://github.com/<username>/<repository>.git
cd fake-news-detection-nlp
```

### Build

```bash
./compile.sh
```

Or:

```bash
mvn clean package -DskipTests
```

### Run

```bash
./run.sh
```

Or:

```bash
java -jar target/fake-news-detector-1.0.0.jar
```

---

## Demo Walkthrough

### 1. Train Model

Select **Train Model** and choose a dataset and classifier.

The system:

* Loads the dataset
* Splits the data into training and testing sets
* Builds the TF-IDF vocabulary
* Trains the classifier
* Evaluates the model
* Saves the trained model

### 2. Evaluate Model

Select **Evaluate Model** to view:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
```

### 3. Predict News

Enter a headline and article body.

Example:

```text
Enter News Headline:
NASA Mars Rover Discovers Evidence of Ancient Lake Bed

Enter News Body:
Scientists analyzing data from NASA's rover reported
findings indicating the presence of ancient water flows.
```

The system produces:

```text
VERDICT       : [REAL NEWS]
CONFIDENCE    : 66.54%

TOP CONTRIBUTING TERMS:

rover       +0.1716  -> REAL
water       +0.0505  -> REAL
report      +0.0655  -> REAL
```

---

## Sample Output

### REAL News

```text
============================================================
                    PREDICTION RESULT
============================================================

VERDICT       : [REAL NEWS]
CONFIDENCE    : 66.54%
PROBABILITY   : 66.54% Real | 33.46% Fake
TOKENS PARSED : 15

TOP CONTRIBUTING TERMS:

rover       +0.1716  -> REAL
mar         +0.0858  -> REAL
find        +0.0698  -> REAL
report      +0.0655  -> REAL
water       +0.0505  -> REAL
```

### FAKE News

```text
============================================================
                    PREDICTION RESULT
============================================================

VERDICT       : [FAKE NEWS]
CONFIDENCE    : 72.58%
PROBABILITY   : 72.58% Fake | 27.42% Real
TOKENS PARSED : 16

TOP CONTRIBUTING TERMS:

secret          -0.2090  -> FAKE
underground    -0.1250  -> FAKE
reveal         -0.1154  -> FAKE
whistleblow    -0.1012  -> FAKE
shock          -0.0812  -> FAKE
```

---

## Research Paper Comparison

The referenced paper reported the following results:

| Algorithm           | Paper Accuracy | Implemented |
| ------------------- | -------------: | :---------: |
| Passive-Aggressive  |     **97.86%** |     Yes     |
| Logistic Regression |     **96.65%** |     Yes     |
| Random Forest       |         95.81% |      No     |
| Decision Tree       |         95.29% |      No     |

> **Note:** These accuracy values are from the research paper and are not the performance results of the sample dataset included in this repository.

---

## Project Structure

```text
fake-news-detection-nlp/
├── compile.sh
├── run.sh
├── pom.xml
├── README.md
├── data/
│   ├── sample_news.csv
│   └── dataset_instructions.md
├── models/
│   └── trained_model.bin
└── src/
    └── main/
        └── java/
            └── com/
                └── fakenews/
                    ├── Main.java
                    ├── data/
                    ├── feature/
                    ├── ml/
                    ├── nlp/
                    └── util/
```

---

## Research Paper Reference

Shaik, M. A., Sree, M. Y., Vyshnavi, S. S., Ganesh, T., Sushmitha, D., & Shreya, N. (2023).  
**Fake News Detection using NLP.**  
2023 International Conference on Innovative Data Communication Technologies and Application (ICIDCA), 399–405.

**DOI:** https://doi.org/10.1109/ICIDCA56705.2023.10100305

---

## Student Details

| Field        | Details                         |
| ------------ | ------------------------------- |
| **Name**     | *Aiden Peter Rodrigues*                   |
| **Roll No.** | *5024153*            |
| **Branch**   | *B.Tech Information Technology* |
