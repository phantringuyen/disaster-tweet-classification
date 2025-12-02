def build_tfidf(corpus, max_features=3000):
    tfidf = TfidfVectorizer(max_features=max_features, min_df=5, max_df=0.85, sublinear_tf=True)
    X = tfidf.fit_transform(corpus)
    return tfidf, X

def save_tfidf(tfidf, path='results/tfidf_vectorizer.joblib'):
    joblib.dump(tfidf, path)

def load_tfidf(path='results/tfidf_vectorizer.joblib'):
    return joblib.load(path)