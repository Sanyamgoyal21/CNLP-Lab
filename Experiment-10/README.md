# Experiment 10 – Low-Resource Neural Machine Translation (WMT 2026 Indic MT)

**Aim:** Build and evaluate a **neural machine translation (NMT)** system for **low-resource Indic languages** using **transfer learning**. A pretrained multilingual Transformer, **NLLB-200**, is fine-tuned on the WMT 2026 shared-task data, compared with its zero-shot (not fine-tuned) performance, and served through a **Flask web application**.

---

## Dataset

**Source:** [WMT 2026 Shared Task on Low-Resource Indic Language Translation](https://www2.statmt.org/wmt26/indic-mt-task.html), 2026 training data (public Google Drive folder, one `.csv` / `.xlsx` file per language pair, English ⇄ X).

| Key | Language | WMT26 category | Script | Clean sentence pairs* | In NLLB-200 pre-training? |
|---|---|---|---|---|---|
| `as` | **Assamese** (used for the results) | I | Bengali-Assamese | 52,580 | yes |
| `lus` | Mizo | I | Latin | 49,741 | yes |
| `mni` | Manipuri | I | Bengali | 23,005 | yes |
| `kha` | Khasi | I | Latin | 25,924 | no |
| `njz` | Nyishi | I | Latin | 50,112 | no |
| `brx` | Bodo | II | Devanagari | 12,768 | no |
| `tgj` | Tagin | II | Latin | 5,203 | no |
| `trp` | Kokborok | II | Latin | 2,265 | no |
| `nag` | Nagamese | II | Latin | 1,997 | no |
| `mjw` | Karbi | II | Latin | 960 | no |

\* After the cleaning steps described below.

English–**Meitei Mayek** is not included: the NLLB vocabulary has no Meitei Mayek characters, so about 47% of its tokens become `<unk>`.

**Split:** each language is shuffled with a fixed seed (42), and **500 pairs** (or 10% for tiny datasets) are held out as the **test set**, saved to `data/<lang>/test.tsv`. The official 2026 test set is only available to registered teams.

---

## Implementation Steps

### 1. Data download and loading (`nmt.py`)
- `gdown` downloads the language file from the public WMT26 Drive folder.
- The files have different layouts, so the loader:
  - reads the first non-empty sheet and drops near-empty columns
  - **finds the English column automatically** by counting common English function words (the Tagin file has English in the *second* column)
  - **detects and drops header rows** ("English | Assamese", "en | as"); the Mizo and Nagamese files have no header.

### 2. Preprocessing
- Unicode NFC normalisation, removal of non-breaking and zero-width spaces, whitespace collapsing.
- Re-attach tokenised punctuation ("soil ." → "soil.").
- **Filtering:** remove empty pairs, `nan` cells, lines longer than 150 words, likely misaligned pairs (length ratio > 3.5 when both sides have more than 3 words), and duplicates.

### 3. Model: NLLB-200 (`facebook/nllb-200-distilled-600M`)
- A 600M-parameter Transformer **encoder-decoder** pretrained on 200 languages, with a shared SentencePiece vocabulary of 256k.
- Language tags select the direction: `eng_Latn` and `asm_Beng`, `lus_Latn`, `mni_Beng`. For languages NLLB has never seen, the tag of a related language in the same script is reused (`lus_Latn` for Latin-script languages, `hin_Deva` for Bodo), and the model learns the new language during fine-tuning.

### 4. Fine-tuning (transfer learning)
- **One bidirectional model** per language pair: every training pair becomes an English→X example *and* an X→English example, each tagged with its source and target language codes.
- `Seq2SeqTrainer` settings: Adafactor optimizer (low memory), learning rate 1e-4, 5% warm-up, weight decay 0.01, batch size 16, max length 128 tokens, fp16 on GPU, 1 epoch.

### 5. Evaluation
- Beam search with 4 beams in both directions on the held-out test set.
- **BLEU** (word n-gram precision), **chrF++** (character n-grams plus word bigrams, more reliable for morphologically rich languages) and **TER** (edit rate, lower is better), computed with `sacrebleu`. **COMET** is optional (`--comet`).
- The **zero-shot NLLB-200** baseline is evaluated on the same test set before fine-tuning.

### 6. Web application (`app.py`)
A Flask server that **imports `nmt.py`**; no translation logic is duplicated. It provides:
- a language-pair selector, direction swap (⇄), and model choice (auto / fine-tuned / zero-shot)
- sentence-by-sentence translation that keeps line breaks
- the evaluation results table and sample translations, read from `results/<lang>.json`
- a JSON API: `POST /api/translate`, `GET /api/languages`.

---

## How to Run

### Install
```bash
pip install -r requirements.txt
```

### Train on a GPU (Google Colab T4)
```bash
!pip install -q -r requirements.txt
!python nmt.py train --lang as --max-train-samples 20000 --epochs 1
```
This downloads the data, evaluates the zero-shot baseline, fine-tunes the model, evaluates it, and saves `models/nllb-as/` and `results/as.json`. If the GPU logs "OOM" warnings, use `--batch-size 8 --grad-accum 2`.

Copy `models/nllb-as/` and `results/as.json` to the same locations on your computer, for example through Google Drive. If Drive renames `model.safetensors` to `model-001.safetensors`, rename it back.

### Other commands
```bash
python nmt.py languages                                   # list all language pairs
python nmt.py evaluate  --lang as --model finetuned       # or --model base
python nmt.py translate --lang as --direction en2x "Education is the key to a better future."
```

### Run the website (works on a CPU)
```bash
python app.py            # then open http://127.0.0.1:5000
```
Assamese, Mizo and Manipuri work even without fine-tuning (zero-shot NLLB-200). The other languages must be fine-tuned first. The first request loads the model (about 2.4 GB); after that, translations take about 2–5 s on a CPU.

---

## Results – English ⇄ Assamese

**Training:** 20,000 sentence pairs (40,000 examples in both directions), 1 epoch, 2,500 steps, **29.0 min on a Colab T4**. The training loss fell from 2.24 to about 1.70.

**Test set (500 held-out pairs):**

| Direction | Model | BLEU ↑ | chrF++ ↑ | TER ↓ |
|---|---|---|---|---|
| English → Assamese | zero-shot NLLB-200 | 8.68 | 32.13 | 91.44 |
| | **fine-tuned NLLB-200** | **14.43** (+5.75) | **37.90** (+5.77) | **81.75** (−9.69) |
| Assamese → English | zero-shot NLLB-200 | 20.66 | 41.48 | 79.38 |
| | **fine-tuned NLLB-200** | **27.79** (+7.13) | **46.16** (+4.68) | **69.25** (−10.13) |

**Sample translations (fine-tuned model):**

| Source | Translation |
|---|---|
| I love you | মই তোমাক ভাল পাওঁ |
| How are you? | তুমি কেনে আছা? |
| The farmers of this village grow rice and vegetables. | এই গাওঁখনৰ কৃষকসকলে চাউল আৰু শাক পাচলি খেতি কৰে। |
| Our development partnerships reflect the development priorities of our partner nations. | আমাৰ উন্নয়নমূলক অংশীদাৰীত্বই আমাৰ অংশীদাৰ দেশসমূহৰ উন্নয়নৰ অগ্রাধিকাৰসমূহক প্ৰতিফলিত কৰে। |
| মই তোমাক ভাল পাওঁ। (Assamese → English) | I love you. |

---

## Analysis

- **Fine-tuning helps in both directions.** BLEU rose by about 6–7 points and TER fell by about 10 after just one epoch on 20,000 pairs. Transfer learning from a large multilingual model works well when parallel data is limited.
- **Assamese → English scores higher than English → Assamese** (27.8 vs 14.4 BLEU). NLLB generates English much better, and Assamese is **morphologically rich**: a word with one wrong suffix counts as a whole wrong word in BLEU. chrF++ gives partial credit at the character level, so the gap there is smaller.
- **The training data is noisy, a typical low-resource problem.** In one test pair the "reference" Assamese sentence is an unrelated Bible verse (*Micah 6:8*), not a translation of the English sentence. Such misaligned pairs lower both the training quality and the measured scores, which is why the cleaning filters matter.
- **Limitations:** a single epoch on 20,000 of the 52,000 pairs, BLEU computed with the default tokenizer, and the test split taken from the training release rather than the official WMT test set. More data and epochs, and checking scores with COMET, are the next steps.

---

## Files

```
Experiment-10/
├── nmt.py               # NMT core: download, cleaning, split, fine-tuning, evaluation, translation, CLI
├── app.py               # Flask website (imports nmt.py)
├── templates/index.html # web page
├── static/app.js        # page logic (calls /api/translate)
├── static/style.css     # styling (light/dark)
├── requirements.txt
├── results/as.json      # scores, training log, sample translations
├── data/                # downloaded data + splits (git-ignored)
└── models/nllb-as/      # fine-tuned model, ~2.4 GB (git-ignored)
```
