# Experiment 4 – Term Frequency, Named Entity Recognition and TF-IDF

**Aim:** Measure how important words are in a text. This experiment computes **term frequency** with and without an NLP toolkit, identifies **named entities** with NLTK, and implements **TF-IDF from scratch**.

---

## Tasks

| No. | Task |
|---|---|
| 4.1 | With a toolkit (NLTK): word frequencies saved to CSV (`Term, Frequency`), top 10 terms, and Named Entity Recognition |
| 4.2 | The same term-frequency analysis **without any toolkit** |
| 4.3 | Without NLTK/spaCy: read multiple documents, then compute TF, DF, IDF and TF-IDF and the top terms per document |

---

## Dataset

| Input | Used in | Description |
|---|---|---|
| `4.1_4.2_input.txt` | 4.1, 4.2 | English article about NLP, about 2,400 words (mentions Indian languages, Python, NLP tasks) |
| 3 short documents (in the notebook) | 4.3 | D1: *"Natural language processing is a field of artificial intelligence."* D2: *"Natural language processing helps computers understand human language."* D3: *"Machine learning is an important part of artificial intelligence."* |

---

## Implementation Steps

### 4.1 Term frequency and NER with NLTK
1. **Preprocess.** Lowercase, tokenize with `word_tokenize`, and keep only alphabetic tokens (`word.isalpha()`).
2. **Count.** `collections.Counter(words)`.
3. **Save.** Write every `(term, frequency)` pair to `term_frequency_4_1.csv`.
4. **Top 10.** `frequency.most_common(10)`.
5. **NER.** Tokenize the *original* (cased) text, run `pos_tag`, then `ne_chunk`. Every `Tree` chunk is a named entity, with a label such as PERSON, ORGANIZATION or GPE.

### 4.2 Term frequency without a toolkit
1. Lowercase, then tokenize with the regex `\b[a-z]+\b`.
2. Count with `Counter`, save to `term_frequency_without_toolkit.csv` sorted by frequency, and print the top 10.

### 4.3 TF-IDF from scratch
1. **Preprocess each document.** Lowercase, remove punctuation (`[^\w\s]`), split on whitespace.
2. **TF** = (count of the term in the document) / (total terms in the document).
3. **DF** = number of documents that contain the term.
4. **IDF** = `log(N / DF)` (natural log, N = 3).
5. **TF-IDF** = TF × IDF.
6. Print Term, TF, DF, IDF and TF-IDF at each step, and the top 10 TF-IDF terms per document.

---

## How to Run

1. Open `CNLP_Experiment_4_.ipynb` in Google Colab or Jupyter.
2. Place `4.1_4.2_input.txt` next to the notebook (or in Drive) and update the path
   (currently `/content/drive/MyDrive/CLNLP-Lab/Experiment-4/4.1_4.2_input.txt`).
3. `pip install nltk`. The notebook downloads `punkt`, `averaged_perceptron_tagger_eng`, `maxent_ne_chunker_tab` and `words`.
4. **Run all.** Two CSV files are created in the working directory.

---

## Results

**Top 10 terms:**

| Rank | 4.1 NLTK | Freq. | 4.2 No toolkit | Freq. |
|---|---|---|---|---|
| 1 | a | 88 | a | 90 |
| 2 | and | 86 | and | 86 |
| 3 | the | 75 | the | 75 |
| 4 | to | 42 | of | 42 |
| 5 | in | 42 | to | 42 |
| 6 | can | 42 | in | 42 |
| 7 | of | 40 | can | 41 |
| 8 | is | 37 | is | 37 |
| 9 | may | 30 | may | 30 |
| 10 | word | 30 | word | 30 |

**NER (NLTK `ne_chunk`), sample:** `NLP → ORGANIZATION`, `English → GPE`, `Assamese → GPE`, `Tamil → GPE`, `Telugu → GPE`, `Hindi → PERSON`, `Bengali → PERSON`, `Marathi → PERSON`, `Python → GPE`.

**4.3 TF-IDF (top terms):**

| Document | Highest TF-IDF terms |
|---|---|
| D1 | `a`, `field` (0.1221), then the shared terms (0.0451) |
| D2 | `helps`, `computers`, `understand`, `human` (0.1373), `language` (0.1014) |
| D3 | `machine`, `learning`, `an`, `important`, `part` (0.1221) |

Terms that occur in 2 of the 3 documents (`natural`, `language`, `processing`, `artificial`, `intelligence`, `is`, `of`) have IDF = log(3/2) = 0.4055, so their TF-IDF is low.

---

## Observations

- Raw term frequency is dominated by **function words** (`a, and, the`), so it says little about the topic until stop words are removed. TF-IDF fixes this by down-weighting terms that appear in many documents.
- **NLTK and regex counts differ slightly** (`a`: 88 vs 90, `of`: 40 vs 42, `can`: 42 vs 41) because the two tokenizers split contractions, hyphens and apostrophes differently.
- **NLTK's NER makes clear errors**: language names are tagged PERSON or GPE (`Hindi → PERSON`) and `Python → GPE`. The rule-based chunker relies on capitalization and has no notion of "language" as a category; a statistical model such as spaCy does better.
- In 4.3, `language` occurs twice in D2, which gives it a higher TF (0.25) and a higher TF-IDF than the other shared words.

---

## Files

```
Experiment-4/
├── CNLP_Experiment_4_.ipynb                   # notebook (4.1, 4.2, 4.3)
├── 4.1_4.2_input.txt                          # input text for 4.1 and 4.2
└── _Experiment-4_ Term Frequency and NER.docx # problem statement
```
Generated when run: `term_frequency_4_1.csv`, `term_frequency_without_toolkit.csv`.
