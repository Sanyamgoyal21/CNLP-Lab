# Experiment 3 – Stemming, Lemmatization and Regular Expressions

**Aim:** Reduce words to their base forms with **stemming** (Porter Stemmer) and **lemmatization** (WordNet Lemmatizer), and use **regular expressions** to extract structured information (emails, URLs, phone numbers, hashtags, mentions) from free text.

---

## Tasks

| No. | Task |
|---|---|
| 3.1 | Stem 8 given words with the Porter Stemmer and display word → stem |
| 3.2 | Lemmatize 14 given words with the WordNet Lemmatizer |
| 3.3 | Extract email addresses, URLs, mobile numbers, hashtags and mentions with regex |

---

## Dataset (inputs given in the problem statement)

- **3.1 words:** `playing, played, plays, studies, studying, connected, connection, computers`
- **3.2 words:** `cats, dogs, running, runs, ran, studies, studying, better, children, mice, went, ate, leaves, caring`
- **3.3 text:** a 7-line workshop announcement containing 2 emails, 3 URLs, 2 mobile numbers (`+91-9876543210`, `9123456789`), 3 hashtags and 2 mentions.

---

## Implementation Steps

### 3.1 Stemming
1. Create `nltk.stem.PorterStemmer()`.
2. Apply `stemmer.stem(word)` to each word. The Porter algorithm strips suffixes using rewrite rules such as `-ing`, `-ed`, `-ies → i` and `-ion`.

### 3.2 Lemmatization
1. Download the `wordnet` and `omw-1.4` resources.
2. Create `WordNetLemmatizer()`.
3. **Give the part of speech.** WordNet lemmatization depends on it:
   - nouns (`cats, dogs, children, mice, leaves`) → `pos="n"`
   - adjective (`better`) → `pos="a"`
   - all remaining words → `pos="v"` (verb)
4. Print word → lemma.

### 3.3 Regex extraction
| Item | Pattern used |
|---|---|
| Email | `[\w.-]+@[\w.-]+\.\w+` |
| URL | `https?://\S+\|www\.\S+` |
| Mobile number | `(?:\+91[- ]?)?[6-9]\d{9}` (optional +91, 10 digits starting 6–9) |
| Hashtag | `#\w+` |
| Mention | `@\w+` |

---

## How to Run

1. Open `CLNLP_Experiment_3.ipynb` in Google Colab or Jupyter.
2. Install NLTK if needed: `pip install nltk`. The notebook downloads WordNet itself.
3. **Run all.** No data files are needed; all inputs are in the notebook.

---

## Results

**3.1 Porter Stemmer:**

| Word | Stem | | Word | Stem |
|---|---|---|---|---|
| playing | play | | studying | studi |
| played | play | | connected | connect |
| plays | play | | connection | connect |
| studies | studi | | computers | comput |

**3.2 WordNet Lemmatizer:**

| Word | Lemma | | Word | Lemma |
|---|---|---|---|---|
| cats | cat | | better | good |
| dogs | dog | | children | child |
| running | run | | mice | mouse |
| runs | run | | went | go |
| ran | run | | ate | eat |
| studies | study | | leaves | leaf |
| studying | study | | caring | care |

**3.3 Regex extraction:**

| Item | Extracted |
|---|---|
| Emails | `nlpworkshop@gmail.com`, `support@python.org` |
| URLs | `https://www.nlpworkshop.com`, `www.python.org.`, `https://github.com/NLPWorkshop` |
| Mobile numbers | `+91-9876543210`, `9123456789` |
| Hashtags | `#NLP`, `#Python`, `#MachineLearning` |
| Mentions | `@gmail`, `@python`, `@NLPWorkshop`, `@PythonLearner` |

---

## Observations

- **Stemming vs lemmatization:** stemming is fast but can produce non-words (`studi`, `comput`). Lemmatization returns real dictionary words and handles irregular forms (`went → go`, `mice → mouse`, `better → good`), but needs the correct part of speech.
- **Regex limitations seen in the output:**
  - The mention pattern also matched the domain part of the email addresses (`@gmail`, `@python`). A lookbehind such as `(?<![\w.])@\w+` would avoid this.
  - The URL `www.python.org.` keeps the sentence-ending full stop. Stripping trailing punctuation would fix it.

---

## Files

```
Experiment-3/
├── CLNLP_Experiment_3.ipynb   # notebook (3.1, 3.2, 3.3)
└── Experiment-3.docx          # problem statement
```
