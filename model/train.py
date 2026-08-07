from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "model" / "data" / "cars_prepared.csv"
ARTIFACT_PATH = ROOT_DIR / "model" / "artifacts" / "cars_pipe.pkl"


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(
        include=["int32", "int64", "float32", "float64"]
    ).columns.tolist()
    categorical_features = x.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def candidate_models() -> list:
    models = [
        LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
        RandomForestClassifier(
            max_depth=4,
            min_samples_leaf=2,
            min_samples_split=2,
            random_state=42,
        ),
    ]

    if XGBClassifier is not None:
        models.append(
            XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
        )

    return models


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    x = df.drop(columns=["price_category"])
    y = df["price_category"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    preprocessor = build_preprocessor(x)

    best_score = 0.0
    best_pipe = None

    for estimator in candidate_models():
        pipe = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", estimator),
            ]
        )

        try:
            scores = cross_val_score(pipe, x, y_encoded, cv=4, scoring="accuracy")
            print(
                f"model: {type(estimator).__name__}, "
                f"accuracy_mean: {scores.mean():.4f}, "
                f"accuracy_std: {scores.std():.4f}"
            )

            if scores.mean() > best_score:
                best_score = scores.mean()
                best_pipe = pipe

        except Exception as exc:
            print(f"Skipped {type(estimator).__name__}: {exc}")

    if best_pipe is None:
        raise RuntimeError("No model was trained successfully.")

    best_pipe.fit(x, y_encoded)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": best_pipe,
            "label_encoder": label_encoder,
            "metadata": {
                "name": "car price category prediction pipeline",
                "author": "Belousov Danila",
                "version": "1.0.0",
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "model_type": type(best_pipe.named_steps["model"]).__name__,
                "cv_accuracy": round(float(best_score), 4),
                "target": "price_category",
            },
        },
        ARTIFACT_PATH,
    )

    print(
        f"best_model: {type(best_pipe.named_steps['model']).__name__}, "
        f"cv_accuracy: {best_score:.4f}"
    )
    print(f"saved artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()

