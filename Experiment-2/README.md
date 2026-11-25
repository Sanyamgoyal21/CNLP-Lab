# Experiment 2 – Basic Text Preprocessing, Tokenization and Stop-Word Removal

**Aim:** Implement the first steps of every NLP pipeline: cleaning raw text, splitting it into sentences and words, and removing stop words. Tokenization is done three ways (**NLTK**, **spaCy**, and **plain Python / regex**) so they can be compared.

---

## Tasks

| No. | Task |
|---|---|
| 2.1 | Basic preprocessing: lowercase, remove punctuation, numbers and extra whitespace, and count each removed item |
| 2.2 | Word and sentence tokenization: (a) NLTK, (b) spaCy, (c) without NLTK/spaCy |
| 2.3 | Tokenize a text, remove common stop words, and display the original tokens, the stop words and the filtered tokens |

---

## Dataset

| File | Used in | Content |
|---|---|---|
| `2.1_text_data.txt` | 2.1 | Short English paragraph about NLP with capitals, punctuation (`. , ( ) + !`), numbers (e.g. "2026") and extra spaces |
| `2.2_tokenization_data.txt` | 2.2 | Multi-sentence paragraph about NLP and its applications |
| `2.3_clean_data.txt` | 2.3 | Paragraph about NLP used for stop-word removal |

---

## Implementation Steps

### 2.1 Text preprocessing
1. **Lowercase.** Count the uppercase characters with `ch.isupper()`, then apply `text.lower()`.
2. **Remove punctuation.** Collect the characters found in `string.punctuation`, report the distinct types, then remove them with `str.translate`.
3. **Remove numbers.** Find them with the regex `\d+`, count them, then remove them with `re.sub`.
4. **Remove extra whitespace.** Measure runs of 2 or more whitespace characters (`\s{2,}`), then collapse them to one space with `re.sub(r'\s+', ' ', text)`.
5. **Display the cleaned text.**

### 2.2 Tokenization (three implementations)
| Method | Sentence tokenizer | Word tokenizer |
|---|---|---|
| NLTK | `sent_tokenize` (Punkt model) | `word_tokenize` (Treebank rules) |
| spaCy | `spacy.blank("en")` + `sentencizer` pipe | spaCy rule-based tokenizer (`[t.text for t in doc]`) |
| No library | regex `(?<=[.!?])\s+` | regex `\b[\w-]+\b\|[.,!?;:]` |

### 2.3 Stop-word removal
1. Tokenize the lowercased text with `re.findall(r'\b\w+\b', ...)`.
2. Define a custom stop-word set (`is, the, a, an, in, of, and, to, for, on, with, ...`).
3. Split the tokens into **stop words found** and **filtered tokens**, and print the original, stop-word and filtered lists.

---

## How to Run

1. Open `CLNP_Experiment_2.ipynb` in Google Colab or Jupyter.
2. Keep the three `.txt` files next to the notebook (or in Google Drive) and update the `open(...)` paths. The notebook uses `/content/drive/MyDrive/CNLP-Lab/Experiment-2/...` in some cells and `.../CLNLP-Lab/...` in others, so check each one.
3. Install the libraries: `pip install nltk spacy`. The NLTK cell downloads `punkt` and `punkt_tab` itself.
4. **Runtime → Run all.**

---

## Results

**2.1 Preprocessing statistics (`2.1_text_data.txt`):**

| Check | Result |
|---|---|
| Uppercase letters | 32 |
| Distinct punctuation types | 6 (`. , ( ) + !`) |
| Numbers removed | 3 |
| Extra whitespace characters removed | 12 |

Cleaned text (start): *"natural language processing nlp is a branch of artificial intelligence it helps computers understand and process human language in nlp is used in many applications such as chatbots machine translation …"*

**2.2 Tokenization:** all three methods split the paragraph into the same sentences, for example:
1. Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence.
2. It enables computers to understand human language.
3. NLP is used in many real-world applications, such as chatbots, machine translation, sentiment analysis, and text summarization.

The word tokenizers differ on punctuation and hyphens. NLTK splits off `(`, `)`, `.` and `,` as separate tokens and keeps `real-world` together, and the regex version keeps hyphenated words through `[\w-]+`.

**2.3 Stop words:** words such as `is, an, of, it, in, such, as, and, the, to, a, can` are removed. The filtered list keeps the content words: `natural, language, processing, important, field, artificial, intelligence, used, many, applications, chatbots, …`.

---

## Observations

- Removing numbers and punctuation shortens the text but loses information. For example, "In 2026" becomes "in", so preprocessing should match the task.
- The NLTK and spaCy tokenizers handle abbreviations and punctuation better than a simple regex. The regex version is transparent and needs no downloads.
- A custom stop-word list is easy to control, but it has to be extended for each domain.

---

## Files

```
Experiment-2/
├── CLNP_Experiment_2.ipynb     # notebook (2.1, 2.2 a/b/c, 2.3)
├── 2.1_text_data.txt
├── 2.2_tokenization_data.txt
├── 2.3_clean_data.txt
└── Experimemt-2.docx           # problem statement
```
