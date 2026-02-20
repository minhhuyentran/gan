from sklearn.svm import OneClassSVM

def train_ocsvm(X_train, nu=0.05, kernel="rbf", gamma="scale"):
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_train)
    return model

def score_ocsvm(model, X):
    # sklearn: decision_function higher = more normal
    s = model.decision_function(X).ravel()
    return s
