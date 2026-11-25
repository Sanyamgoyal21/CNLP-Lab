# Experiment 9 (Mini Project 1) – 30-Second News Reel Script Generator

**Aim:** Build an NLP system that turns a long news article into a **30-second video script** (about 75–85 spoken words) for short-form video creators on Instagram Reels, YouTube Shorts and similar platforms. The script has a **hook, key facts and a call to action**, plus on-screen text cards, hashtags and a caption. It is checked with **ROUGE** and a **fact-preservation score**. The system runs as an interactive **Streamlit** web app.

---

## Key Features

1. **Multi-source news input:** live RSS feeds, any article URL, or pasted text.
2. **NLP preprocessing:** sentence segmentation, named entity recognition, keyword extraction (spaCy).
3. **Topic detection:** six news categories.
4. **Extractive summarization:** graph-based **TextRank** selects the most central sentences.
5. **Reel script generation:** Hook (0–3 s), Key Facts (3–24 s), Call to Action (24–30 s), in four tones: *Informative, Dramatic, Casual, Hinglish*.
6. **Production assets:** on-screen text overlays, hashtags, a ready-to-post caption.
7. **Evaluation:** ROUGE-1/2/L and an entity/number fact-safety audit.

---

## Dataset / Input Sources

There is no fixed training dataset. The system works on **live, real-world news text**:

| Source | Details |
|---|---|
| RSS feeds (`feedparser`) | NDTV (Top Stories, India, Tech/Gadgets360), The Hindu (National, Tech), Times of India (Top, Tech), BBC World News. The latest 5 headlines per feed |
| Article URL | Full article text extracted with **trafilatura**, falling back to **newspaper3k** |
| Direct text | Any pasted article body and headline |
| Test article (`test_pipeline.py`) | A 6-sentence article about an ISRO solar-monitoring satellite mission |

Feed check on 25 Nov 2026: 7 of the 8 feeds returned articles. **NDTV Top Stories** returned none, and the app then shows "No articles returned from selected feed".

---

## Implementation Steps

All NLP logic is in `nlp_pipeline.py`. `app.py` is only the user interface.

### 1. Content ingestion
- `fetch_rss_articles(url)` parses the feed and returns the title, link, summary (HTML tags removed) and publish date.
- `extract_article_text(url)` downloads the page and extracts the main article body without menus, ads or comments. If trafilatura gets fewer than 150 characters, newspaper3k is tried. If both fail, the RSS summary is used.

### 2. Preprocessing (`preprocess_text`)
- Whitespace normalisation.
- spaCy `en_core_web_sm` (first 15,000 characters):
  - **Sentence segmentation:** sentences shorter than 10 characters are dropped.
  - **Named entities:** `(text, label)` pairs such as ORG, GPE, PERSON, DATE, CARDINAL.
  - **Keywords:** lemmas of nouns, proper nouns and adjectives that are not stop words and are longer than 2 characters. The 12 most frequent are kept.
- If spaCy is unavailable, the code falls back to regex sentence splitting and a small stop-word list.

### 3. Topic detection (`detect_topic`)
Keyword matching (`\b…\b` regex) against six categories: *Technology & AI, Space & Science, Politics & Policy, Business & Economy, Sports, Entertainment*. The category with the most matches wins, or *General News* if none match.

### 4. Extractive summarization with TextRank (`extractive_textrank`)
1. Build a graph with one node per sentence.
2. Edge weight = word-overlap similarity: sim(Sᵢ, Sⱼ) = |Sᵢ ∩ Sⱼ| / (log(|Sᵢ|+1) + log(|Sⱼ|+1)).
3. Run **PageRank** (damping factor 0.85, 25 iterations) over the row-normalised similarity matrix.
4. Take the top-*k* sentences (*k* = 2–5, default 3), returned in their original order.

### 5. Reel script generation (`generate_reel_script`)
- **Clean** sentences by removing datelines and agency tags (`New Delhi:`, `(PTI)`, `Reuters`, …).
- **Hook:** a tone-specific opening template that includes the detected topic.
- **Body:** the TextRank sentences joined with spoken connectors ("First, … Second, … In addition, …", or Hinglish connectors).
- **Closer:** a tone-specific call to action.
- **Length control:** if the script exceeds *target + 15* words, the body is shortened.
- **Duration estimate:** words ÷ **2.7 words per second** (normal speaking pace).

### 6. Production assets (`generate_social_assets`)
Four on-screen text cards (topic, key point, context, action), hashtags (the topic tag, 4 generic news tags and the top 4 keywords), and a caption.

### 7. Evaluation
- **ROUGE-1 / ROUGE-2 / ROUGE-L** (`rouge_score`, with stemming): the generated script scored against the TextRank summary. This measures how much of the selected factual content the script keeps.
- **Fact Safety Index** (`verify_facts`): the percentage of the article's first 15 named entities that appear in the script, plus the list of numbers it contains. This guards against facts being dropped or changed.

### 8. Web interface (`app.py`)
A Streamlit app with three input tabs (RSS / URL / Direct text). The sidebar sets tone, target words (60–100) and the number of extracted sentences. The output shows a metrics row, a colour-coded teleprompter, production assets, and evaluation tabs (ROUGE, TextRank sentences, entity audit).

---

## How to Run

```bash
cd Experiment-9
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python test_pipeline.py          # run the pipeline tests
streamlit run app.py             # start the web app -> http://localhost:8501
```

Notes:
- **Internet is needed** for the RSS and URL inputs. Direct text works offline.
- **`newspaper3k` needs `lxml_html_clean`** with recent `lxml` versions. It is listed in `requirements.txt`.
- If `pip install` upgrades `protobuf` or `starlette` and breaks other packages in the same environment, use a virtual environment or `pip install "streamlit>=1.30,<1.41"`.
- Deployment options (Streamlit Cloud, Render, Docker, Linux VPS) are described in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Results

Output of `python test_pipeline.py` on the ISRO test article (Informative tone, 3 sentences, 80-word target). All tests passed.

| Step | Output |
|---|---|
| Sentences detected | 6 |
| Keywords | space, isro, launch, satellite, mission, earth, indian, research, organisation, … |
| Entities | *The Indian Space Research Organisation* (ORG), *Sunday* (DATE), *ISRO* (ORG), *over 500* (CARDINAL), *the past three years* (DATE), … |
| Topic | **Space & Science** |
| Script length | **82 words → 30.4 seconds** |
| ROUGE-1 (P / R / F1) | 0.702 / 1.000 / **0.825** |
| ROUGE-2 (P / R / F1) | 0.675 / 0.966 / **0.794** |
| ROUGE-L (P / R / F1) | 0.702 / 1.000 / **0.825** |
| Fact Safety Index | 42.9% (6 of 14 entities kept) |
| Hashtags | #SpaceScience #NewsBrief #CurrentAffairs #Journalism #DailyUpdate #Space #Isro #Launch #Satellite |

**Generated script:**

> **Hook:** Here is your 30-second briefing on the latest in Space & Science.
> **Key facts:** First, The Indian Space Research Organisation (ISRO) on Sunday announced the launch date for its next-generation solar monitoring satellite. Second, Scientists confirmed that the mission aims to study coronal mass ejections and space weather phenomena that affect global communication networks. In addition, Launch preparations are in full swing at the Satish Dhawan Space Centre in Sriharikota, with liftoff scheduled for next month.
> **Call to action:** Follow for verified, bite-sized news breakdowns every day.

---

## Analysis

- **TextRank picked the right sentences:** the announcement, the mission goal and the launch preparations. It left out supporting details (number of engineers, interest from other agencies).
- **The script fits the 30-second format:** 82 words ≈ 30.4 s at 2.7 words per second.
- **ROUGE recall is 1.0** because the script keeps the extracted sentences word for word. Precision is about 0.70 because the hook and closer add template words. This is the expected behaviour of an **extractive + template** approach: the facts are preserved, but the wording is not paraphrased.
- **The Fact Safety Index (42.9% = 6 of 14 entities) is a coverage measure.** The 6 kept entities are ISRO and its full name, *Sunday*, *Satish Dhawan Space Centre*, *Sriharikota* and *next month*. All 8 missing ones (*over 500*, *L1*, *1.5 million kilometers*, *NASA*, *ESA*, …) come from the 3 sentences TextRank left out. Nothing in the script contradicts the source, but a 30-second script cannot carry every fact.
- **Limitations:** topic detection is keyword-based, hooks are templates, and RSS/URL extraction depends on each site's markup. An abstractive model such as mT5 or mBART (Experiment 8) could make the script more natural.

---

## Project Structure

```
Experiment-9/
├── app.py              # Streamlit UI: input tabs, teleprompter, assets, evaluation
├── nlp_pipeline.py     # NLP logic: ingestion, preprocessing, NER, TextRank, script, ROUGE, fact check
├── test_pipeline.py    # end-to-end pipeline test
├── requirements.txt    # Python dependencies
├── DEPLOYMENT.md       # hosting guide
└── README.md
```
