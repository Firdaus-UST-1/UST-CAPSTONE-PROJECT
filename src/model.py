from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_model(X_train, y_train, preprocessor) -> Pipeline:
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model

