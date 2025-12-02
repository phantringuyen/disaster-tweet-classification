# disaster-tweet-classification

# Tweet Disaster Classification — Project (Modular + Colab)

This project folder contains Python source files and a Colab notebook for the Disaster Tweet Classification datathon. The structure, scripts, and README are below. Use the notebook for interactive exploration and the `src/` scripts for CLI runs.

---

## Folder structure

```
tweet-disaster-classification/
│
├── notebooks/
│   └── tweet_classification_colab.ipynb   # Colab-ready notebook (placeholder)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train_logreg.py
│   ├── train_svm.py
│   ├── train_random_forest.py
│   ├── train_naive_bayes.py
│   ├── train_distilbert.py
│   ├── train_modernbert.py
│   └── utils.py
│
├── data/
│   ├── README.md   # instructions to place train.csv & test.csv here
│
├── results/
│   ├── plots/
│   └── model_checkpoints/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Key files (templates)

### `src/data_preprocessing.py`

```python
"""Preprocessing utilities: cleaning, tokenization, saving cleaned CSVs."""
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"\bhttp\b|\bhttps\b|\btco\b|\bco\b|\bamp\b", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train.csv')
    parser.add_argument('--output', type=str, default='data/train_clean.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['clean_text'] = df['text'].astype(str).apply(clean_text)
    df.to_csv(args.output, index=False)
    print('Saved:', args.output)
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
if __name__ == '__main__':
    df = pd.read_csv('data/train_clean.csv')
    X = df['clean_text']
    y = df['target']
    tfidf, X_all = build_tfidf(X)
    save_tfidf(tfidf)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print(classification_report(y_val, preds))
    joblib.dump(model, 'results/model_logreg.joblib')
```

### `src/train_svm.py`, `train_random_forest.py`, `train_naive_bayes.py`

* Similar to `train_logreg.py` but with `LinearSVC`, `RandomForestClassifier`, and `MultinomialNB` respectively. Keep consistent tfidf load/save.

### `src/train_distilbert.py`

```python
# fine-tuning DistilBERT with Trainer (uses cleaned text CSV)
DF = pd.read_csv('data/train_clean.csv')
X = DF['clean_text'].tolist()
y = DF['target'].tolist()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=160)

train_ds = Dataset.from_dict({'text': X_train, 'labels': y_train}).map(tokenize, batched=True)
val_ds   = Dataset.from_dict({'text': X_val, 'labels': y_val}).map(tokenize, batched=True)

train_ds.set_format('torch', columns=['input_ids','attention_mask','labels'])
val_ds.set_format('torch', columns=['input_ids','attention_mask','labels'])

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

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
trainer.save_model('results/model_distilbert')
```

### `src/train_modernbert.py`

* Same as `train_distilbert.py` but using `ModernBertForSequenceClassification` and its tokenizer (`answerdotai/ModernBERT-base`).

### `src/utils.py`

* small helpers: save/load models, plot confusion matrix, show random samples.

---

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

## .gitignore

```
__pycache__/
.ipynb_checkpoints/
*.pyc
results/
data/*.csv
```

---
