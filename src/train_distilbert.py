import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, ModernBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

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