from sklearn.ensemble import IsolationForest

def train_iforest(X_train, contamination=0.05, n_estimators=300, random_state=42):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train)
    return model

def score_iforest(model, X):
    # higher = more normal (score_samples)
    return model.score_samples(X)
