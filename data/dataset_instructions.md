# Dataset Instructions

This application supports the two datasets analyzed in the research paper:

### Option 1: Kaggle `news.csv` (George McIntire Dataset - 6,335 articles)
* **Columns**: `title`, `text`, `label` (`REAL` / `FAKE`)
* **Download**: From Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) or [Fake News Detection Dataset](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news).
* **Usage**:
  1. Download and rename to `news.csv`.
  2. Place it inside this `data/` folder (`data/news.csv`).
  3. When running the application, choose Option `1` (Train Model) and enter:
     ```text
     data/news.csv
     ```

### Option 2: ISOT Fake News Dataset (44,898 articles)
* **Files**: `True.csv` (21,417 rows) and `Fake.csv` (23,481 rows)
* **Columns**: `title`, `text`, `subject`, `date`
* **Usage**:
  The system automatically extracts the `title` and `text`, concatenates them into `article`, and discards the `date` and `subject` metadata exactly as described in Section III-D of the paper.

### Option 3: Built-in Sample Dataset
* A curated 30-article test set is preloaded in `data/sample_news.csv` so you can train, evaluate, and test predictions immediately without downloading gigabytes of data.
