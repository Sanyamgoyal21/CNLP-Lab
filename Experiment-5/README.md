# Experiment 5 – Subword Tokenization (BPE, SentencePiece), POS Tagging and Word2Vec

**Aim:** Break text into **subword units** with Byte Pair Encoding and SentencePiece, both with pretrained tokenizers and with tokenizers trained from scratch. Then tag words with their **part of speech** using spaCy and NLTK, and learn **word embeddings** with Word2Vec (CBOW and Skip-Gram).

---

## Tasks

| No. | Task |
|---|---|
| 5.1 | BPE subword tokenization: (a) pretrained (GPT-2), (b) implemented from scratch. Display tokens and token IDs |
| 5.2 | SentencePiece tokenization: (a) pretrained (T5), (b) trained on our own data |
| 5.3 | POS tagging of *"The young student is reading an interesting book in the library."*: (a) spaCy, (b) NLTK. Display each unique token, its tag and the tag description |
| 5.4 | POS tagging of the full input text with spaCy, plus the frequency of each unique token |
| 5.5 | Train a Word2Vec **CBOW** model |
| 5.6 | Train a Word2Vec **Skip-Gram** model |
| 5.7 | Cosine similarity between word pairs |

---

## Dataset

| File | Used in | Description |
|---|---|---|
| `input_sub_word_data.txt` | 5.1, 5.2, 5.4 | English text about NLP, 947 words / 6,682 characters |
| `corpus.txt` | 5.5–5.7 | 55-word toy corpus of short sentences ("Natural language processing is interesting", "I love machine learning", …) |
| `my_sentencepiece.model` / `.vocab` | 5.2 b | SentencePiece model **produced by** the notebook |

---

## Implementation Steps

### 5.1 BPE
**(a) Pretrained.** `GPT2TokenizerFast.from_pretrained("gpt2")` → `encode()` gives the token IDs, `convert_ids_to_tokens()` gives the tokens, and `decode()` checks the round trip. (`Ġ` marks a leading space and `Ċ` a newline.)

**(b) From scratch:**
1. Split the text into words and punctuation (regex), lowercase, and represent each word as characters plus the end-of-word marker `</w>`.
2. Count the frequency of every adjacent symbol pair across the vocabulary.
3. Merge the most frequent pair into a new symbol, and repeat for **100 merges**.
4. Tokenize text by applying the learned merges in order, then map each unique token to an ID (ID 0 is reserved for unknown tokens).

### 5.2 SentencePiece
**(a) Pretrained.** `T5Tokenizer.from_pretrained("t5-small")` (SentencePiece unigram model, `▁` marks a word start).

**(b) Trained from scratch:**
```python
spm.SentencePieceTrainer.train(input="input_sub_word_data.txt", model_prefix="my_sentencepiece",
                               vocab_size=100, model_type="unigram",
                               character_coverage=1.0, hard_vocab_limit=False)
```
The trained model is then loaded with `SentencePieceProcessor` to encode the text into pieces and IDs and decode it back.

### 5.3 POS tagging
- **spaCy.** `en_core_web_sm` gives universal tags (`token.pos_`), and `spacy.explain()` gives the descriptions. Punctuation is skipped and unique tokens are kept.
- **NLTK.** `word_tokenize` then `pos_tag` gives Penn Treebank tags (`DT, JJ, NN, VBZ, VBG, …`), and a dictionary maps each tag to its description.

### 5.4 POS tagging with frequencies
spaCy processes the full input file, a `Counter` counts the lowercased tokens, and a DataFrame shows *Unique Token, POS Tag, Description, Frequency*.

### 5.5–5.7 Word2Vec (gensim)
Each line of `corpus.txt` is lowercased and split into words, then:
```python
Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=0, epochs=100, seed=42)  # CBOW
Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=1, epochs=100, seed=42)  # Skip-Gram
```
`model.wv[word]` gives a vector, `model.wv.most_similar(word)` the nearest words, and `model.wv.similarity(a, b)` the cosine similarity.

---

## How to Run

1. Keep `input_sub_word_data.txt` and `corpus.txt` in the same folder as `CLNLP_Experiment_5.ipynb` (the notebook uses relative paths).
2. Install the libraries:
   ```bash
   pip install transformers sentencepiece spacy nltk gensim pandas
   python -m spacy download en_core_web_sm
   ```
3. **Run all.** The first run downloads the GPT-2 and T5 tokenizers. Section 5.2 b writes `my_sentencepiece.model` and `my_sentencepiece.vocab`.

---

## Results

**Subword tokenization of `input_sub_word_data.txt`:**

| Method | Pretrained? | Vocabulary | Number of tokens |
|---|---|---|---|
| GPT-2 BPE | Yes | ~50k | **1,348** |
| BPE from scratch | No | 122 symbols (100 merges) | 4,227 |
| T5 SentencePiece | Yes | ~32k | **1,319** |
| SentencePiece from scratch (unigram) | No | 100 pieces | 3,950 |

The first BPE merges learned from scratch were `e + </w>`, `s + </w>`, `e + n` and `i + n`. These are the most common English word endings and letter pairs. GPT-2's byte-level BPE decodes back to the **identical** text (`decoded_text == text` → `True`). T5 decodes the same words, but its SentencePiece normalisation turns line breaks into spaces.

**POS tagging (5.3):**

| Token | spaCy | NLTK |
|---|---|---|
| The | DET (determiner) | DT (Determiner) |
| young | ADJ (adjective) | JJ (Adjective) |
| student | NOUN | NN |
| is | AUX | VBZ |
| reading | VERB | VBG |
| an | DET | DT |
| interesting | ADJ | JJ |
| book | NOUN | NN |
| in | ADP (adposition) | IN |
| library | NOUN | NN |

**5.4:** the full input text has **1,209 spaCy tokens** and **368 unique tokens**. The most frequent are punctuation (`,` = 76, `.` = 65), followed by function words.

**Word2Vec (5.5–5.7):**

| Query | Result |
|---|---|
| CBOW: most similar to `machine` | world (0.249), deep (0.219), useful (0.201) |
| Skip-Gram: most similar to `language` | processing (0.092), interesting (0.077), and (0.063), helps (0.053), natural (0.052) |
| similarity(machine, learning) | 0.1606 |
| similarity(natural, language) | 0.0473 |
| similarity(deep, learning) | 0.2592 |

---

## Observations

- **Pretrained tokenizers are about 3× more efficient.** They produce roughly 1,300 tokens versus about 4,000 for the from-scratch models, because their large vocabularies hold whole words such as `Ġlanguage`. A 100-merge or 100-piece vocabulary splits most words into small pieces.
- **Subword tokenization avoids out-of-vocabulary words:** any new word can still be built from characters and subwords.
- **spaCy and NLTK agree** on this sentence. They only use different tag sets (Universal vs Penn Treebank), and NLTK's is finer-grained (`VBZ` vs `VBG`).
- **The Word2Vec similarities are low and noisy** (all below 0.3) because `corpus.txt` has only 55 words. Embeddings need large corpora; compare the pretrained GloVe vectors in Experiment 6.

---

## Files

```
Experiment-5/
├── CLNLP_Experiment_5.ipynb    # notebook (5.1 – 5.7)
├── input_sub_word_data.txt     # input text for 5.1, 5.2, 5.4
├── corpus.txt                  # toy corpus for Word2Vec
├── my_sentencepiece.model      # SentencePiece model trained in 5.2 b
├── my_sentencepiece.vocab      # its 100-piece vocabulary
└── Experiment-5.docx           # problem statement
```
