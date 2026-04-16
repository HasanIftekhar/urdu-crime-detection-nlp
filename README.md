# Crime Detection in Urdu Audio (NLP Pipeline)

**CASE Institute of Technology — Final Year Project (FYP)**

## Overview

Built an end-to-end NLP pipeline for sentiment-based crime detection in Urdu — a low-resource language with no pre-existing labeled dataset. Collected and processed a self-generated corpus of **7,000 audio clips (20 hours)**, achieving **89% speech-to-text accuracy**, and applied Word2Vec sentence embeddings with K-Means clustering to classify criminal vs. non-criminal speech.

## Pipeline

```
Urdu Audio → Speech-to-Text (Google STT) → Text Preprocessing → 
Word2Vec Embeddings → K-Means Clustering → Crime Classification
```

## Key Results

- **89%** speech-to-text accuracy on Urdu audio
- Self-collected corpus: 7,000 clips / 20 hours of audio
- Unsupervised clustering separating criminal vs. non-criminal speech
- Silhouette score used for optimal cluster validation

## Tech Stack

- Python
- Google Speech-to-Text API (`speech_recognition`)
- Gensim (Word2Vec — Urdu Wikipedia vectors: 140M tokens, 100K vocab, 300 dimensions)
- Scikit-learn (K-Means, Silhouette analysis)
- UrduHack (Urdu NLP preprocessing)
- NLTK
- Tkinter (GUI)

## Word2Vec Model

The Urdu Word2Vec model (`urduvec_140M_100K_300d.bin`, ~119MB) is not included due to file size.

Download from: [Urdu Word Vectors — URDUVEC](https://github.com/jaleed96/urduvec)

Place the `.bin` file in the `clustering/` folder before running.

## Dataset

| File | Description |
|------|-------------|
| `Roman Urdu DataSet.csv` | Romanized Urdu text samples |
| `Urdu Abusive Dataset.csv` | Labeled abusive/criminal Urdu phrases |
| `Crime.csv` | Crime classification labels |
| `clustering/SpeechToText.csv` | STT output for training |
| `clustering/SpeechToText_TESTING.csv` | STT output for testing |

## Files

| File | Description |
|------|-------------|
| `Fyp Code.py` | Main STT pipeline — audio processing, translation, crime word detection |
| `Sentiment Analysis.py` | NLP preprocessing — tokenization, lemmatization, Word2Vec |
| `main.py` | Word2Vec model loading and vector testing |
| `clustering/Clustring.py` | K-Means clustering on Word2Vec embeddings |
| `clustering/Shihouette.py` | Silhouette score analysis for cluster validation |
| `clustering/Front.py` | GUI frontend (Tkinter) |


## How to Run

```bash
pip install SpeechRecognition gensim scikit-learn urduhack nltk googletrans tinytag

# Test Word2Vec embeddings
python main.py

# Run full STT + classification pipeline
python "Fyp Code.py"

# Run clustering
python clustering/Clustring.py
```

> Requires Google Speech API access and the Urdu Word2Vec `.bin` model in `clustering/`.
