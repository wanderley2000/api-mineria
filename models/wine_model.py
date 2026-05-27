import requests
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score

# ── Estado global del modelo ──────────────────────────────────────────────────
model_state = {
    "model": None,
    "scaler": None,
    "model_name": None,
    "features": [],
    "precision": None,
    "accuracy": None,
    "params": {},
    "trained": False,
}

ALL_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)


def load_dataset(sample_size: float = 1.0) -> pd.DataFrame:
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text), sep=";")
    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)
    if sample_size < 1.0:
        df = df.sample(frac=sample_size, random_state=42).reset_index(drop=True)
    return df


def train_model(
    model_name: str = "random_forest",
    features: list = None,
    test_size: float = 0.2,
    scale: bool = True,
    sample_size: float = 1.0,
    # params específicos por modelo
    n_estimators: int = 100,
    max_depth: int = None,
    n_neighbors: int = 5,
) -> dict:
    if features is None:
        features = ALL_FEATURES

    df = load_dataset(sample_size)
    X = df[features]
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        params = {"n_estimators": n_estimators, "max_depth": max_depth}

    elif model_name == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        params = {"max_depth": max_depth}

    elif model_name == "knn":
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        params = {"n_neighbors": n_neighbors}

    else:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    precision = round(precision_score(y_test, preds), 4)
    accuracy = round(accuracy_score(y_test, preds), 4)

    model_state["model"] = clf
    model_state["scaler"] = scaler
    model_state["model_name"] = model_name
    model_state["features"] = features
    model_state["precision"] = precision
    model_state["accuracy"] = accuracy
    model_state["params"] = params
    model_state["trained"] = True

    return {
        "model": model_name,
        "features_used": features,
        "precision": precision,
        "accuracy": accuracy,
        "params": params,
        "test_size": test_size,
        "scale": scale,
        "sample_size": sample_size,
    }
