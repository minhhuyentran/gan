from sklearn.svm import OneClassSVM

def train(X_train, nu=0.05, kernel="rbf", gamma="scale"):
    m = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    m.fit(X_train)
    return m

def score(model, X):
    # higher => more normal
    return model.decision_function(X).ravel()
