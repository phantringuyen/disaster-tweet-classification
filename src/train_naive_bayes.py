if __name__ == '__main__':
    df = pd.read_csv('data/train_clean.csv')
    X = df['clean_text']
    y = df['target']
    tfidf, X_all = build_tfidf(X)
    save_tfidf(tfidf)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print(classification_report(y_val, preds))
    joblib.dump(model, 'results/model_naive_bayes.joblib')