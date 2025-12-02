if __name__ == '__main__':
    df = pd.read_csv('data/train_clean.csv')
    X = df['clean_text']
    y = df['target']
    tfidf, X_all = build_tfidf(X)
    save_tfidf(tfidf)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=200,   # number of trees
        max_depth=30,       # prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print(classification_report(y_val, preds))
    joblib.dump(model, 'results/model_random_forest.joblib')