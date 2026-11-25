# Experiment 6 – GloVe Word Embeddings and N-gram Language Model

**Aim:**
1. Implement **GloVe (Global Vectors for Word Representation)** to learn word embeddings from a text corpus and find the words most similar to **"king"**. This is done twice: (a) **from scratch**, without any pretrained model, and (b) with **pretrained GloVe vectors through Gensim**.
2. Implement a **statistical N-gram language model from scratch** that predicts the next word from the preceding words.

---

## Dataset

| Data | Used in | Description |
|---|---|---|
| `glove_corpus (1).txt` | Part A | 24 short sentences: royalty (king, queen, prince, palace), people (man, woman), professions (doctor, teacher, scientist), animals and nature. 60-word vocabulary after preprocessing. If the file is missing, the notebook uses an identical inline copy |
| `glove-wiki-gigaword-100` (Gensim downloader) | Part A.2 | Pretrained GloVe: Wikipedia 2014 + Gigaword 5, 6B tokens, **400,000 words**, 100 dimensions (about 128 MB) |
| N-gram corpus (in the notebook) | Part B | 8 sentences given in the problem statement, e.g. *"natural language processing is a branch of artificial intelligence"*. 35-word vocabulary |

---

## Implementation Steps

### Part A.1 – GloVe from scratch (NumPy)
1. **Preprocess.** Lowercase, tokenize with the regex `[a-z]+`, and remove frequent function words (`the, a, an, is, and, in, of, at, to, around`). On a 24-sentence corpus these words co-occur with everything and would dominate the vectors.
2. **Vocabulary.** 60 unique words, mapped `word2idx` / `idx2word`.
3. **Co-occurrence matrix X (60 × 60).** Symmetric window of size 3, where a context word at distance *d* adds **1/d**. This gives 214 non-zero entries.
4. **Model.** Word vectors *wᵢ*, context vectors *w̃ⱼ* and biases *bᵢ, b̃ⱼ*, 30 dimensions. The model minimises the GloVe weighted least-squares loss

   J = Σ f(Xᵢⱼ) · (wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ − log Xᵢⱼ)²,  with f(x) = (x / x_max)^α for x < x_max, otherwise 1 (x_max = 5, α = 0.75)
5. **Training.** AdaGrad (learning rate 0.05), 1,000 epochs over the non-zero Xᵢⱼ entries.
6. **Embedding.** The final vector for each word is *wᵢ + w̃ᵢ*. Neighbours are found with **cosine similarity**, and the vectors are projected to 2-D with PCA (via SVD) for plotting.

### Part A.2 – Pretrained GloVe (Gensim)
1. `gensim.downloader.load("glove-wiki-gigaword-100")`.
2. `most_similar("king")`, the analogies *king − man + woman* and *prince − man + woman*, and pairwise similarities.
3. Neighbours of "king" restricted to the corpus vocabulary, a PCA plot, and a side-by-side comparison with the scratch model.

### Part B – N-gram language model from scratch
1. **Preprocess.** Split the corpus into sentences on `.`, lowercase, tokenize, and pad each sentence with `n−1` start tokens `<s>` and one end token `</s>`.
2. **Count.** For n = 1, 2, 3, count every context → next-word occurrence (`defaultdict(Counter)`).
3. **Probability with add-k smoothing (k = 0.01).** P(w | h) = (C(h, w) + k) / (C(h) + k·|V|), so unseen N-grams never get zero probability.
4. **Next-word prediction.** Rank the vocabulary by P(w | last n−1 words) and return the top 3.
5. **Generation.** Repeatedly append the most probable word (greedy decoding) until `</s>`.
6. **Evaluation.** Perplexity = exp(−(1/N) Σ log P(wᵢ | h)) on the training sentences and on 3 new test sentences.

---

## How to Run

1. Open `CLNLP_Experiment_6.ipynb` in Jupyter, VS Code or Colab, with `glove_corpus (1).txt` in the same folder (optional, since an inline copy exists).
2. Requirements: `numpy pandas matplotlib gensim`. The notebook runs `pip install gensim` itself.
3. **Run all.** Part A.2 needs internet to download the 128 MB GloVe vectors the first time. Everything else runs on a CPU in under a minute.

---

## Results

### Part A – Words most similar to "king"

The from-scratch training loss fell from 5.67 (epoch 1) to 0.0005 (epoch 1000).

| Rank | GloVe from scratch (24 sentences) | Pretrained GloVe (6B tokens) |
|---|---|---|
| 1 | son (0.97) | prince (0.77) |
| 2 | powerful (0.94) | queen (0.75) |
| 3 | queen (0.93) | son (0.70) |
| 4 | prince (0.90) | brother (0.70) |
| 5 | wise (0.90) | monarch (0.70) |

| Pair | Scratch | Pretrained |
|---|---|---|
| king – queen | 0.932 | 0.751 |
| king – prince | 0.902 | 0.768 |
| king – palace | −0.962 | 0.538 |
| king – cat | 0.140 | 0.328 |
| king – computer | 0.197 | 0.198 |

**Analogies (pretrained):**
- *king − man + woman* = **queen** (0.770), then monarch, throne, daughter, princess.
- *prince − man + woman* = **princess** (0.751).

**Other scratch neighbours:** cat → dog (0.94), man → woman (0.89), computer → processes (0.99), sun → gives (1.00).

### Part B – N-gram language model

| Model | Train perplexity | Test perplexity |
|---|---|---|
| Unigram | 26.642 | 28.349 |
| Bigram | 2.071 | **3.204** |
| Trigram | **1.746** | 3.902 |

| Input | Bigram prediction | Trigram prediction |
|---|---|---|
| natural language | processing (0.410) | processing (0.899) |
| artificial | intelligence (0.899) | intelligence (0.748) |
| machine learning is | a (0.899) | a (0.855) |
| deep learning is a | branch (0.376) | branch (0.600) |

| Seed | Generated by the trigram model |
|---|---|
| natural | natural language processing uses language models learn patterns from text data |
| a language | a language model predicts the next word in a sentence |
| artificial intelligence includes | artificial intelligence includes machine learning and deep learning |

---

## Observations

- **The scratch GloVe model learns corpus-specific relations.** King, queen, prince and son share contexts such as *rules*, *kingdom* and *powerful*. With only 214 co-occurrence counts it overfits (the loss approaches 0) and produces unreliable values, such as a strongly negative king–palace similarity.
- **The pretrained GloVe vectors capture real semantics** (prince, monarch, throne) and solve analogies. Embedding quality depends heavily on corpus size.
- **More context lowers training perplexity:** from 26.6 (unigram) to 2.07 (bigram) to 1.75 (trigram).
- **The trigram model is worse on the test sentences** (3.90 vs 3.20) because of **data sparsity**: many test trigrams were never seen, and smoothing gives them very small probabilities.
- **Greedy bigram generation can loop.** "natural" produced *"natural language processing uses language processing uses …"*, while the trigram's longer context generates fluent training-like sentences.

---

## Files

```
Experiment-6/
├── CLNLP_Experiment_6.ipynb   # Part A.1, A.2 and B (with saved outputs)
├── glove_corpus (1).txt       # GloVe training corpus
└── Experiment-6 (1).docx      # problem statement
```
