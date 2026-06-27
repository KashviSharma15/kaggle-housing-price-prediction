# kaggle-housing-price-prediction

# Comment Category Prediction Challenge
**Kaggle Competition | IIT Madras BS in Data Science & Applications**

---

## Problem Statement

This is a **multi-class text classification** competition where the goal is to categorize online comments into one of **4 toxicity/sentiment categories** (labels 0–3) based on the comment text and associated metadata such as upvotes, downvotes, emoticons, and demographic flags (race, religion, gender, disability).

The dataset contains **198,000 training samples** and **102,000 test samples**, making it a large-scale NLP challenge with real-world noise and missing values.

---

## Dataset

| File | Rows | Columns |
|---|---|---|
| `train.csv` | 198,000 | 15 (including label) |
| `test.csv` | 102,000 | 14 (no label) |
| `Sample.csv` | — | Submission format |

### Key Columns

| Column | Description |
|---|---|
| `comment` | Raw text of the online comment (primary feature) |
| `upvote` / `downvote` | Community engagement signals |
| `emoticon_1/2/3` | Emoticon presence flags |
| `race`, `religion`, `gender` | Demographic context flags (with missing values) |
| `disability` | Boolean disability flag |
| `post_id` | Post identifier |
| `label` | Target — 0 (non-toxic), 1, 2, 3 (varying toxicity levels) |

### Label Distribution (Submission)
| Label | Count | Meaning |
|---|---|---|
| 0 | 56,464 | Non-toxic / Neutral |
| 2 | 30,888 | Moderately toxic |
| 1 | 10,002 | Mildly toxic |
| 3 | 4,646 | Highly toxic |

---

## Approach & Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed label distribution — highly imbalanced (class 0 dominates at ~57%)
- Plotted comment length distribution — right-skewed with most comments being short
- Examined missing values in `race`, `religion`, `gender` columns
- Computed upvote/downvote patterns across label classes

### 2. Text Preprocessing
- Converted all comments to string, handled `NaN` values
- Applied **TF-IDF Vectorization** with:
  - `max_features = 5000`
  - `stop_words = 'english'`
  - `min_df = 5`, `max_df = 0.8`

### 3. Feature Engineering
- `comment_length` — character count of each comment
- `word_count` — number of words in each comment
- Filled missing `race`, `religion`, `gender` with `'Unknown'`
- Converted `disability` to integer
- Combined TF-IDF text features with structured metadata using `ColumnTransformer` + `Pipeline`

### 4. Models Trained & Compared

| Model | Validation Accuracy |
|---|---|
| Dummy Classifier (Baseline) | 57.66% |
| Logistic Regression (untuned) | 77.87% |
| Logistic Regression (tuned) | **78.01%** |
| SGD Classifier (untuned) | 74.88% |
| SGD Classifier (tuned) | 76.20% |
| Naive Bayes | 71.45% |
| KNN (with TruncatedSVD) | 62.17% |
| SVM — LinearSVC | 78.19% |
| SVM + Chi² Feature Selection | 78.19% |
| Bagging Classifier | 75.94% |
| LightGBM (Boosting) | 77.92% |
| MLP Neural Network | 74.33% |
| **Stacking (NB + SVM → LR)** | **78.22%** |

### 5. Hyperparameter Tuning
- Used **GridSearchCV** with 3-fold cross-validation
- Best Logistic Regression: `C=10`, `solver='lbfgs'`, `max_iter=500`
- Best SGD: `alpha=0.0001`, `penalty='l1'`

### 6. Final Model — Stacking Ensemble
The best performing model was a **Stacking Classifier** combining:
- **Base learners:** Naive Bayes + LinearSVC
- **Meta-learner:** Logistic Regression
- Achieved **78.22% validation accuracy**

---

## Results Summary

```
Best Model        : Stacking Classifier (NB + SVM → LR)
Validation Accuracy: 78.22%
Baseline Accuracy : 57.66%
Improvement       : +20.56% over baseline
Dataset Size      : 198,000 train | 102,000 test
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| NLP / Text | Scikit-learn TF-IDF, TruncatedSVD, SelectKBest |
| ML Models | Scikit-learn, LightGBM |
| Tuning | GridSearchCV |
| Visualization | Matplotlib, Seaborn |
| Environment | Kaggle Notebooks (GPU disabled) |

---

## Project Structure

```
kaggle-comment-category-prediction/
│
├── Kashvi-notebook.ipynb    # Main Kaggle notebook with all code
├── README.md                # This file
└── submission_final.csv     # Final predictions (generated at runtime)
```

---

## How to Run

1. Clone this repository
2. Upload `train.csv` and `test.csv` from the competition to `/kaggle/input/competitions/comment-category-prediction-challenge/`
3. Run all cells in `Kashvi-notebook.ipynb` sequentially
4. The final `submission_final.csv` will be saved to `/kaggle/working/`

---

## 🔍 Key Learnings

- **TF-IDF + SVM** is a very strong baseline for text classification, often outperforming complex neural models on structured NLP tasks
- **Class imbalance** (57% class 0) significantly inflates accuracy — precision/recall per class is more informative
- **Stacking** provides marginal but consistent gains over individual models when base learners make different types of errors
- Adding **structured metadata** (upvotes, emoticons, demographic flags) alongside text features improves classification of edge cases

---

## Author

**Kashvi Sharma**
IIT Madras BS in Data Science & Applications
Kaggle ID: 24F1001565
