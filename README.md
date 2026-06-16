# 📱 Sentiment Analysis of YouTube Comments on iPhone 17 Launch Rumors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)
[![Published](https://img.shields.io/badge/Published-JOISIE%20Vol.9%20No.2%202025-green)](paper/iphone17-sentiment.pdf)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **Published Research** · JOISIE (Journal of Information Systems and Informatics Engineering), Vol. 9, No. 2, December 2025  
> DOI: [10.35145/joisie.v9i2.5684](https://doi.org/10.35145/joisie.v9i2.5684)

---

## 📖 Overview

This project presents an **end-to-end text mining pipeline** that analyzes Indonesian public sentiment toward the iPhone 17 using YouTube comments scraped before the product's official launch. It compares three machine learning classification algorithms — **Naive Bayes, Random Forest, and K-Nearest Neighbor (KNN)** — to determine the best-performing model for Indonesian social media text.

### Key Findings

| Metric | Result |
|--------|--------|
| Total Comments Collected | 1,077 |
| Positive Sentiment | **43.0%** (460 comments) |
| Negative Sentiment | **38.8%** (415 comments) |
| Neutral Sentiment | **18.1%** (194 comments) |
| Best Model | **Random Forest** |
| Best Model Accuracy | **69.2%** (test) / **65.68%** (10-fold CV) |

The dominant barrier to iPhone 17 adoption in Indonesia was **price sensitivity**, while camera features drove the most positive enthusiasm.

---

## 🗂️ Repository Structure

```
iphone17-sentiment/
│
├── data/
│   ├── dataset_raw_iphone17.csv           # Raw scraped data
│   ├── dataset_cleaned_iphone17.csv       # Cleaned data
│   └── dataset_labeled_iphone17.csv       # Labeled data
│
├── notebooks/
│   ├── 01_crawling_youtube_data.ipynb     # Data collection via YouTube Data API v3
│   ├── 02_data_preprocessing.ipynb        # Cleaning, normalization, stopword, stemming
│   ├── 03_data_labeling.ipynb             # Auto-labeling using InSet Lexicon
│   ├── 04_feature_extraction.ipynb        # TF-IDF vectorization
│   ├── 05_modelling_evaluasi.ipynb        # Model training & evaluation (NB, RF, KNN)
│   └── 06_evaluasi_lanjutan.ipynb         # Cross-validation, confusion matrix, word cloud
│
├── docs/
│   └── iphone17-sentiment.pdf             # Published journal article (JOISIE 2025)
│
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔬 Methodology

The pipeline follows six sequential stages:

```
YouTube API  →  Preprocessing  →  Auto-Labeling  →  TF-IDF  →  Training  →  Evaluation
(Crawling)      (Clean/Stem)      (InSet Lexicon)   (Vectors)   (NB/RF/KNN)  (CV + Metrics)
```

### 1. Data Collection (`01_crawling_youtube_data.ipynb`)
- Scraped using **YouTube Data API v3** with keywords `"iPhone 17 bocoran indonesia"`
- Targeted Indonesian-language tech review channels
- Collected **1,077 comments** from 5 top videos

### 2. Text Preprocessing (`02_data_preprocessing.ipynb`)
- **Cleaning** — removed URLs, mentions, hashtags, punctuation, and emojis using Regex
- **Case Folding** — lowercased all text
- **Normalization** — converted Indonesian slang (*alay*) to formal words using the [colloquial lexicon](https://github.com/nasalsabila/kamus-alay)
- **Stopword Removal** — using the [Sastrawi](https://github.com/har07/PySastrawi) library
- **Stemming** — reduced words to their root form using the Enhanced Confix Stripping (ECS) Stemmer via Sastrawi

### 3. Automatic Labeling (`03_data_labeling.ipynb`)
- Used **InSet Lexicon** (Indonesian Sentiment Lexicon) for polarity scoring
- Scoring rule: `+1` for each positive word, `-1` for each negative word
- Label assignment: **Positive** if score > 0, **Negative** if score < 0, **Neutral** if score = 0

### 4. Feature Extraction (`04_feature_extraction.ipynb`)
- Applied **TF-IDF** (Term Frequency–Inverse Document Frequency) vectorization
- Formula: $W_{i,j} = TF_{i,j} \times \log\left(\frac{N}{DF_i}\right)$

### 5. Modelling (`05_modelling_evaluasi.ipynb`)
- Train/Test split: **80% / 20%** (`random_state=42`)
- Models compared:
  - `MultinomialNB` — Naive Bayes
  - `RandomForestClassifier(n_estimators=100)`
  - `KNeighborsClassifier(n_neighbors=5)`

### 6. Advanced Evaluation (`06_evaluasi_lanjutan.ipynb`)
- **10-Fold Cross Validation** for robust performance estimation
- Confusion matrix analysis
- Word cloud visualization per sentiment class

---

## 📊 Results

### Model Accuracy Comparison

| Algorithm | Test Accuracy | CV Accuracy | Precision | Recall | F1-Score |
|-----------|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **69.2%** | **65.68%** | 0.59 | 0.46 | 0.44 |
| Naive Bayes | 61.2% | 61.55% | 0.63 | 0.44 | 0.42 |
| KNN | 20.6% | 25.91% | 0.32 | 0.33 | 0.16 |

Random Forest achieved the highest accuracy and stability, confirmed by cross-validation. The primary misclassification pattern was **Negative comments predicted as Neutral**, likely due to Indonesian sarcasm and implicit negative expressions (e.g., *"Harganya murah banget ya buat ginjal"* — "The price is so cheap you could sell a kidney") which lexicon-based approaches cannot detect.

### Top Keywords by Sentiment

| Positive | Negative |
|----------|----------|
| iPhone (172), Pro (127), beli/buy (90), lebih/more (69), juta/million IDR (65) | enggak/no (102), harga/price (76), beli/buy (83), mahal/expensive, pajak/tax |

---

## ⚙️ Setup & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Pipeline

Open notebooks in order on **Google Colab** (recommended) or locally with Jupyter:

```bash
jupyter notebook notebooks/01_crawling_youtube_data.ipynb
```

> **Note:** Notebook `01` requires a valid **YouTube Data API v3 key**. Replace the placeholder `API_KEY` in the notebook with your own key from [Google Cloud Console](https://console.cloud.google.com/).

---

## 📦 Dependencies

```
google-api-python-client
pandas
numpy
scikit-learn
Sastrawi
matplotlib
seaborn
wordcloud
```

See [`requirements.txt`](requirements.txt) for pinned versions.

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{modami2025iphone17,
  title     = {Analisis Sentimen Komentar YouTube terhadap Rumor Peluncuran iPhone 17 
               Menggunakan Web Scraping dan Studi Komparatif Algoritma Klasifikasi},
  author    = {Modami, Nickel and Manopo, Ephraim Eleazar Reva and 
               Enditama, Danendra Rafi and Ayunda, Afifah Trista},
  journal   = {JOISIE (Journal Of Information Systems And Informatics Engineering)},
  volume    = {9},
  number    = {2},
  pages     = {403--411},
  year      = {2025},
  doi       = {10.35145/joisie.v9i2.5684}
}
```

---

## 👥 Authors

**Nickel Modami** · **Ephraim Eleazar Reva Manopo** · **Danendra Rafi Enditama** · **Afifah Trista Ayunda**  
Faculty of Science and Technology, Universitas Pradita, Tangerang, Indonesia

---

## 🔮 Future Work

The paper recommends the following improvements for future research:
- Adopt **IndoBERT** or **BERT-based** models to capture semantic context and detect sarcasm
- Integrate **human annotation** as ground truth instead of purely lexicon-based labeling
- Add feature engineering for **emoji patterns** and **punctuation style** to better capture implicit negative sentiment

---

## 📜 License

This project is released under the [MIT License](LICENSE). The published paper is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
