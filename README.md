# Tweet Disaster Classification (From TF-IDF to Transformers)

This project fine-tunes between traditional Machine Learning models and BERT-family models
for Tweet Disaster classification using HuggingFace Transformers.

## Main Pipeline
<img src="https://github.com/phantringuyen/disaster-tweet-classification/blob/main/mainpipeline.png">

## Folder structure

```
tweet-disaster-classification/
│
├── notebooks/
│ └── tweet_classification_colab.ipynb
│
├── src/
│ ├── __init__.py
│ ├── train.py
│ ├── evaluate.py
│ ├── data_preprocessing.py
│ ├── features.py
│ ├── train_logreg.py
│ ├── train_random_forest.py
│ ├── train_naive_bayes.py
│ ├── train_svm.py
│ ├── train_distilbert.py
│ ├── train_bert.py
│ ├── train_modernbert.py
│ ├── train_roberta.py
│ └── utils.py
│
├── data/
│ ├── train.csv
│ ├── test.csv
│
├── results/
│ ├── plots/
│ └── model_checkpoints/
│
├── README.md
├── requirements.txt
└── .gitignore

```

## Contents
- `tweet_disaster_classification.ipynb` — full Colab notebook
- `src/train.py` — training script (HuggingFace Trainer)
- `src/evaluate.py` — evaluation script
- `src/utils.py` — preprocessing + dataset utilities
- `requirements.txt` — package list
- `data/` — dataset folder (ignored by Git)

## How to Run
```
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

## Key files (templates)

### `src/data_preprocessing.py`

```python
"""Preprocessing utilities: cleaning, tokenization, saving cleaned CSVs."""

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"\bhttp\b|\bhttps\b|\btco\b|\bco\b|\bamp\b", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)
```

### `src/features.py` (TF-IDF vectorizer save/load)

```python
def build_tfidf(corpus, max_features=3000):
    tfidf = TfidfVectorizer(max_features=max_features, min_df=5, max_df=0.85, sublinear_tf=True)
    X = tfidf.fit_transform(corpus)
    return tfidf, X

def save_tfidf(tfidf, path='results/tfidf_vectorizer.joblib'):
    joblib.dump(tfidf, path)

def load_tfidf(path='results/tfidf_vectorizer.joblib'):
    return joblib.load(path)
```

### `src/train_logreg.py`

```python
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)
```

### `src/train_svm.py`, `train_random_forest.py`, `train_naive_bayes.py`

* Similar to `train_logreg.py` but with `LinearSVC`, `RandomForestClassifier`, and `MultinomialNB` respectively. Keep consistent tfidf load/save.

### `src/train_distilbert.py`

```python

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results/distilbert',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
```

### `src/train_bert.py`, `src/train_modernbert.py`, `src/train_roberta.py`

* Same as `train_distilbert.py` but using `BertForSequenceClassification`, `ModernBertForSequenceClassification`, `RobertaForSequenceClassification` and its tokenizer.

### `src/utils.py`

* small helpers: save/load models, plot confusion matrix, show random samples, split train and validation dataset.

## requirements.txt

```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
tqdm
joblib
transformers
datasets
torch
wordcloud
```




## References
```
@misc{zisad2025comparativeanalysistransformermodels,
      title={Comparative Analysis of Transformer Models in Disaster Tweet Classification for Public Safety}, 
      author={Sharif Noor Zisad and N. M. Istiak Chowdhury and Ragib Hasan},
      year={2025},
      eprint={2509.04650},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.04650}, 
}
```
```
@misc{nlp-getting-started,
    author = {Addison Howard and devrishi and Phil Culliton and Yufeng Guo},
    title = {Natural Language Processing with Disaster Tweets},
    year = {2019},
    howpublished = {\url{https://kaggle.com/competitions/nlp-getting-started}},
    note = {Kaggle}
}
```
```
@article{journalmedia1010005,
    AUTHOR = {Seddighi, Hamed and Salmani, Ibrahim and Seddighi, Saeideh},
    TITLE = {Saving Lives and Changing Minds with Twitter in Disasters and Pandemics: A Literature Review},
    JOURNAL = {Journalism and Media},
    VOLUME = {1},
    YEAR = {2020},
    NUMBER = {1},
    PAGES = {59--77},
    URL = {https://www.mdpi.com/2673-5172/1/1/5},
    ISSN = {2673-5172},
    DOI = {10.3390/journalmedia1010005}
}
```
```
@inbook{inbook,
    author = {Webb, Geoffrey},
    year = {2016},
    month = {01},
    pages = {1-2},
    title = {Naïve Bayes},
    isbn = {9781489975027},
    doi = {10.1007/978-1-4899-7502-7_581-1}
}
```
```
@inproceedings{inproceedings,
    author = {Liu, Cai-zhi and Sheng, Yan-xiu and Wei, Zhi-qiang and Yang, Yong-Quan},
    year = {2018},
    month = {08},
    pages = {218-222},
    title = {Research of Text Classification Based on Improved TF-IDF Algorithm},
    doi = {10.1109/IRCE.2018.8492945}
}
```
```
@inproceedings{devlin-etal-2019-bert,
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author = "Devlin, Jacob  and
      Chang, Ming-Wei  and
      Lee, Kenton  and
      Toutanova, Kristina",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1423/",
    doi = "10.18653/v1/N19-1423",
    pages = "4171--4186",
}
```
```
@misc{sanh2020distilbertdistilledversionbert,
      title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter}, 
      author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
      year={2020},
      eprint={1910.01108},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1910.01108}, 
}
```
