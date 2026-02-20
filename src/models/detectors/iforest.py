from sklearn.ensemble import IsolationForest

def train(X_train, contamination=0.05, n_estimators=300, random_state=42):
    m = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    m.fit(X_train)
    return m

def score(model, X):
    # higher => more normal
    return model.score_samples(X)
