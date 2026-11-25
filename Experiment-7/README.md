# Experiment 7 – Sentiment Analysis on IMDB: Logistic Regression vs LSTM vs Transformer vs BERT

**Aim:** Classify movie reviews as **positive** or **negative** with four models of increasing complexity, and compare them using **Precision, Recall, F1-score** and **confusion matrices**:

1. **Logistic Regression** on TF-IDF features (classical machine learning)
2. **Bidirectional LSTM** trained from scratch (recurrent neural network)
3. **Vanilla Transformer encoder** trained from scratch (self-attention)
4. **BERT** (`bert-base-uncased`, pretrained, then fine-tuned)

---

## Dataset

| Property | Value |
|---|---|
| Name | IMDB Dataset of 50K Movie Reviews ([Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)) |
| Columns | `review` (text), `sentiment` (positive / negative) |
| Size | 50,000 reviews, **balanced** (25,000 positive / 25,000 negative), no missing values |
| Duplicates | 418 duplicate reviews (removed), leaving 49,582 |
| Review length | mean 231 words, median 173, max 2,470 |
| Split | stratified 80/20: **39,665 train / 9,917 test**. The **same test set is used for all four models** |

Loading is automatic. The notebook uses `IMDB Dataset.csv` if it's present, otherwise downloads it with `kagglehub`, and falls back to the Hugging Face `imdb` dataset.

---

## Implementation Steps

### 1. Preprocessing
- Remove HTML tags (`<br />`) and URLs, keep only letters and apostrophes, collapse whitespace, lowercase.
- Labels: positive → 1, negative → 0.
- Example: *"A wonderful little production. `<br /><br />`The filming technique is very unassuming- very old-time-BBC fashion…"* becomes *"a wonderful little production the filming technique is very unassuming very old time bbc fashion…"*

### 2. Model 1 – Logistic Regression
- `TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)` turns the reviews into a 39,665 × 50,000 sparse matrix.
- `LogisticRegression(C=4.0, max_iter=1000)`.

### 3. Shared setup for the neural models (PyTorch)
- Word vocabulary of the 20,000 most frequent training words (0 = padding, 1 = unknown). Each review is cut or padded to **256 tokens**.
- 10% of the training data is held out for validation. Batch size 64, Adam optimizer, binary cross-entropy loss, gradient clipping at 1.0. The epoch with the best validation accuracy is kept.

### 4. Model 2 – Bidirectional LSTM (3.2M parameters)
`Embedding(20000, 128)` → **2-layer BiLSTM** (128 hidden units each way, dropout 0.4, packed sequences so padding is ignored) → final forward + backward states (256) → Dropout → Linear(256, 1). 5 epochs, learning rate 1e-3.

### 5. Model 3 – Vanilla Transformer encoder (2.9M parameters)
Token embedding (128) + **learned positional embedding** → **2 Transformer encoder layers** (4-head self-attention, feed-forward 256, residual connections + LayerNorm, dropout 0.1) with a padding mask → masked mean pooling → Linear(128, 1). 5 epochs, learning rate 5e-4.

### 6. Model 4 – BERT fine-tuning
- `bert-base-uncased` + classification head on the `[CLS]` token (`AutoModelForSequenceClassification`, 2 labels).
- Raw reviews (only HTML removed) go through the WordPiece tokenizer, max length 256.
- Fine-tuned on **10,000 training reviews** (`BERT_TRAIN_SIZE`, about 25% of the training data, to keep Colab time reasonable) for **2 epochs**: AdamW, lr 2e-5, weight decay 0.01, 10% linear warm-up, batch 16, mixed precision (fp16).

### 7. Evaluation
Accuracy, Precision, Recall and F1 for the positive class, a full classification report, and a confusion-matrix heatmap for each model. Then a comparison table, a bar chart, and predictions on custom reviews.

---

## How to Run

1. Open `CLNLP_Experiment_7.ipynb` in **Google Colab** and choose **Runtime → Change runtime type → T4 GPU**.
2. **Runtime → Run all.** The first cell installs `kagglehub transformers datasets`, and the dataset downloads automatically (no upload needed).
3. Run time on a T4 is about 15 minutes in total:

   | Model | Training time |
   |---|---|
   | Logistic Regression | ~50 s |
   | LSTM | ~25 s per epoch |
   | Transformer | ~19 s per epoch |
   | BERT | ~134 s per epoch |

   Set `BERT_TRAIN_SIZE = None` to fine-tune BERT on all 39,665 reviews (about 4× longer).

Libraries: `torch`, `transformers`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `kagglehub`.

---

## Results (test set, 9,917 reviews)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression (TF-IDF) | 0.9105 | **0.9032** | 0.9202 | 0.9116 |
| Bidirectional LSTM | 0.8836 | 0.9078 | 0.8549 | 0.8806 |
| Vanilla Transformer | 0.8571 | 0.8579 | 0.8573 | 0.8576 |
| **BERT (fine-tuned)** | **0.9130** | 0.8979 | **0.9327** | **0.9150** |

**Confusion matrices:**

| Model | TN | FP | FN | TP | Total errors |
|---|---|---|---|---|---|
| Logistic Regression | 4,449 | 491 | 397 | 4,580 | 888 |
| LSTM | 4,508 | 432 | 722 | 4,255 | 1,154 |
| Vanilla Transformer | 4,233 | 707 | 710 | 4,267 | 1,417 |
| **BERT** | 4,412 | 528 | **335** | **4,642** | **863** |

**Training curves:**

| Model | Final train accuracy | Final validation accuracy |
|---|---|---|
| LSTM | 0.930 | 0.886 |
| Transformer | 0.938 | 0.860 |
| BERT | 0.947 (after 2 epochs) | — |

**Most informative words (Logistic Regression):**
- positive: *great, excellent, perfect, amazing, wonderful, brilliant, best, hilarious, superb, loved*
- negative: *worst, bad, awful, boring, poor, terrible, waste, horrible, worse, disappointment*

**Custom reviews:**

| Review | LR | LSTM | Transformer | BERT |
|---|---|---|---|---|
| "This movie was absolutely fantastic! …" | positive | positive | positive | positive |
| "Worst film I have ever seen. A complete waste …" | negative | negative | negative | negative |
| "The movie was not bad at all, I actually enjoyed it quite a lot." | positive | **negative** ✗ | **negative** ✗ | positive |
| "I expected a lot more. The plot was predictable…" | negative | negative | **positive** ✗ | negative |

---

## Analysis

- **BERT is the best model:** highest accuracy (91.3%), recall (93.3%) and F1 (0.915), and the fewest errors (863). This is despite being fine-tuned on only 10,000 reviews for 2 epochs, because its pretrained contextual representations already understand English.
- **Logistic Regression is almost as good** (F1 0.912, 888 errors) in under a minute on a CPU, and it has the highest precision. IMDB reviews are long and full of strongly polar words, which a linear TF-IDF model exploits well. It is also the most interpretable model.
- **The LSTM** has high precision (0.908) but lower recall (0.855): it misses more positive reviews (722 false negatives).
- **The Transformer trained from scratch is the weakest** (F1 0.858). Its training accuracy (0.938) is far above its validation accuracy (0.860), so it **overfits**: self-attention without pre-training needs much more data.
- **Negation:** only LR (thanks to bigrams like "not bad") and BERT classify *"not bad at all … enjoyed it"* correctly. The from-scratch sequence models are misled by the word "bad".
- **Trade-off:** BERT gives the best quality at the highest cost (110M parameters, GPU needed). Logistic Regression is the most efficient baseline.

---

## Files

```
Experiment-7/
├── CLNLP_Experiment_7.ipynb   # full pipeline with saved outputs and plots
└── Experiment-7.docx          # problem statement
```
