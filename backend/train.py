from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "backend"
REPORT_DIR = ROOT / "reports"
TARGET = "label"

def main():
    candidates = list(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("Put the PhiUSIIL CSV file inside the project's data/ folder.")
    df = pd.read_csv(candidates[0])
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Available columns: {list(df.columns)}")

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    y_raw = df[TARGET]
    if y_raw.nunique() != 2:
        raise ValueError(f"Expected a binary target, found: {y_raw.unique()}")

    # Exclude identifiers and text fields that cannot be inferred from a numeric feature vector.
    excluded = {TARGET, "URL", "Domain", "TLD", "Title", "FILENAME", "FileName"}
    X = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number", "bool"]).copy()
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    if X.shape[1] == 0:
        raise ValueError("No numeric features found after excluding text/identifier columns.")

    # Keep the dataset's original labels. The API maps label 0 to phishing and 1 to legitimate.
    y = y_raw.astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": list(X.columns)}, MODEL_DIR / "model.pkl")

    report = {
        "rows_used": int(len(df)),
        "features_used": list(X.columns),
        "feature_count": int(X.shape[1]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "target_values": sorted(y.unique().tolist())
    }
    (REPORT_DIR / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "rows_used": report["rows_used"],
        "feature_count": report["feature_count"],
        "accuracy": report["accuracy"],
        "model_file": str(MODEL_DIR / "model.pkl"),
        "report_file": str(REPORT_DIR / "baseline_metrics.json")
    }, indent=2))
    print("\nNote: this random split is only a baseline. Check duplicates, domain leakage, and independent testing before reporting it as reliable.")

if __name__ == "__main__":
    main()
