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