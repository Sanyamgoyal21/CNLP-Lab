# Experiment 8 – Abstractive Text Summarization with mT5 and mBART (XL-Sum, Hindi)

**Aim:** Implement and evaluate **abstractive summarization**, where the model writes a new summary in its own words, using two multilingual Transformer sequence-to-sequence models:

- **mT5:** Multilingual Text-to-Text Transfer Transformer (`google/mt5-small`)
- **mBART-50:** Multilingual Bidirectional and Auto-Regressive Transformer (`facebook/mbart-large-50`)

Both are fine-tuned on the **Hindi** part of the **XL-Sum** dataset and compared with **ROUGE-1, ROUGE-2, ROUGE-L** and **BERTScore (mBERT)**.

---

## Dataset

| Property | Value |
|---|---|
| Name | XL-Sum (Cross-Lingual Summarization), BBC news articles with professionally written summaries ([Hugging Face](https://huggingface.co/datasets/csebuetnlp/xlsum)) |
| Language | Hindi (`hindi` config) |
| Fields | `id`, `url`, `title`, `summary`, `text` |
| Splits | train **70,778**, validation 8,847, test 8,847 |
| Article length | mean 1,091 words (median 1,190) |
| Summary length | mean 36 words (median 34) |
| Used here | **3,000** training articles (random, seed 42) and **200** test articles |

**Loading note:** XL-Sum on the Hub is stored as a loading script (`xlsum.py`), which `datasets` ≥ 4.0 no longer runs. The notebook therefore reads Hugging Face's auto-converted Parquet files directly (`hf://datasets/csebuetnlp/xlsum@refs%2Fconvert%2Fparquet/hindi/...`).

---

## Implementation Steps

### 1. Preprocessing and tokenization
- **Input** = article text, truncated to **512 tokens**. **Target** = reference summary, truncated to **64 tokens**.
- Each model uses its own SentencePiece tokenizer (250k vocabulary). For mBART the source and target language codes are both set to `hi_IN`.
- `DataCollatorForSeq2Seq` pads each batch dynamically and masks label padding with −100, so it is ignored in the loss.

### 2. Model development (fine-tuning with `Seq2SeqTrainer`)
| Setting | mT5-small | mBART-large-50 |
|---|---|---|
| Parameters | 300M | 611M |
| Pre-training objective | span corruption (mC4, 101 languages) | denoising auto-encoder (text infilling + sentence permutation, 50 languages) |
| Precision | fp32 (mT5 gives NaN losses in fp16) | fp16 mixed precision |
| Batch | 4 × 4 gradient accumulation = 16 | 2 × 8 gradient accumulation = 16 |
| Learning rate | 5e-4 | 5e-5 |
| Optimizer | Adafactor | Adafactor |
| Epochs | 2 | 2 |
| Other | gradient checkpointing, 5% warm-up | gradient checkpointing, 5% warm-up, generation forced to start with the `hi_IN` token |

### 3. Generation
Beam search with 4 beams, `no_repeat_ngram_size=3`, at most 64 new tokens.

### 4. Evaluation metrics
- **ROUGE-1 / ROUGE-2 / ROUGE-L:** F1 overlap of unigrams, bigrams and the longest common subsequence. The default `rouge_score` tokenizer deletes all non-Latin characters, which would give Hindi a score of 0. A **Unicode-aware tokenizer** is used instead: it removes punctuation and symbols (including the danda `।`) but keeps Devanagari letters and vowel signs.
- **BERTScore** with `bert-base-multilingual-cased` (mBERT): semantic similarity of contextual token embeddings (precision, recall and F1).

### 5. Reference baseline
The official checkpoint `csebuetnlp/mT5_multilingual_XLSum` (mT5-base fine-tuned by the XL-Sum authors on all 45 languages) is evaluated on the same 200 articles as an upper bound.

---

## How to Run

1. Open `CLNLP_Experiment_8.ipynb` in **Google Colab** with a **T4 GPU** runtime.
2. **Run all.** The first cell installs `transformers datasets accelerate sentencepiece rouge_score bert_score evaluate`, and the dataset and models download automatically.
3. Settings are in the configuration cell: `LANGUAGE`, `MBART_LANG_CODE`, `TRAIN_SAMPLES`, `TEST_SAMPLES`, `NUM_EPOCHS`, `EVAL_REFERENCE_CHECKPOINT`. For another language, use for example `bengali` / `bn_IN` or `telugu` / `te_IN`.
4. Time on a T4: mT5 fine-tuning **9.9 min**, mBART fine-tuning **18.6 min**, and 45–80 s of generation per model.

**Compatibility fix:** newer `transformers` versions removed the `warmup_ratio` argument. The notebook computes `warmup_steps` instead (5% of the total steps).

---

## Results (200 Hindi test articles)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-P | BERTScore-R | BERTScore-F1 |
|---|---|---|---|---|---|---|
| mT5-small (fine-tuned here) | 26.46 | 7.85 | 20.94 | 74.36 | 71.70 | 72.97 |
| **mBART-large-50 (fine-tuned here)** | **33.65** | **13.80** | **28.12** | **77.36** | **74.83** | **76.02** |
| *mT5-base, official XL-Sum checkpoint (reference)* | *36.76* | *16.70* | *30.33* | *78.60* | *76.07* | *77.27* |

**Average summary length:**

| Summary source | Average length (words) |
|---|---|
| mT5 | 19.1 |
| mBART | 19.6 |
| official mT5 | 20.2 |
| references | 26.1 |

**Example (test article 1, about the Boston Marathon bombing):**

| | Summary |
|---|---|
| Reference | अमरीकी शहर बॉस्टन में पुलिस ने कहा है कि दूसरे ऐसे संदिग्ध व्यक्ति को पकड़ लिया गया है जिस पर … मैराथन के दौरान धमाकों में शामिल होने का शक है… |
| mT5 | **भारत के** बॉस्टन मैराथन में हुए धमाकों के बाद संदिग्धों की तलाश जारी है. |
| mBART | **अमरीका के** बॉस्टन शहर में हुए बम धमाकों के एक संदिग्ध को पकड़ा गया है. |

---

## Analysis

- **mBART outperforms mT5-small on every metric:** +7.2 ROUGE-1, +6.0 ROUGE-2, +7.2 ROUGE-L and +3.1 BERTScore-F1. It is twice as large, and its denoising pre-training (reconstructing whole sentences) matches the generation task closely.
- **mBART is only about 3 ROUGE-1 points behind the official checkpoint,** even though it saw only 3,000 articles for 2 epochs. The official model was trained on the full XL-Sum dataset, so more data would likely close the gap.
- **mT5-small makes more factual errors (hallucinations).** In the example it says Boston is in India ("भारत के बॉस्टन"), while mBART correctly writes "अमरीका के बॉस्टन". In another article mT5 attributes the event to the wrong person (Jaswant Singh instead of his son Manvendra Singh).
- **Both models write shorter summaries** than the references (about 19–20 vs 26 words), which lowers ROUGE recall.
- **ROUGE is low for abstractive summaries** in general because paraphrases don't match word-for-word. BERTScore (73–77) reflects the semantic quality better.
- **Trade-off:** mT5-small trains about twice as fast and needs less memory, while mBART-large gives clearly better and more faithful summaries.

---

## Files

```
Experiment-8/
├── CLNLP_Experiment_8.ipynb   # full pipeline with saved outputs
└── Experiment-8.docx          # problem statement
```
