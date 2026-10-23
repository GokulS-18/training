"""Machine-learning logic for the MPG predictor (no UI code here)."""
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

NUM_COLS = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
CAT_COLS = ["origin", "brand"]
ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}
BRAND_FIX = {"chevroelt": "chevrolet", "chevy": "chevrolet", "maxda": "mazda", "toyouta": "toyota",
             "vokswagen": "volkswagen", "vw": "volkswagen", "mercedes-benz": "mercedes", "capri": "ford"}
MODEL_PATH = "best_mpg_model.joblib"


def load_and_clean(file) -> pd.DataFrame:
    """Read the uploaded CSV, normalise columns, convert '?' to NaN, derive brand."""
    df = pd.read_csv(file, na_values=["?", " ", "NA", "N/A", ""])
    df.columns = [re.sub(r"[\s\-]+", "_", c.strip().lower()) for c in df.columns]
    required = ["mpg"] + [c for c in NUM_COLS + ["origin"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in ["mpg"] + NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # origin may be 1/2/3 or text
    if pd.api.types.is_numeric_dtype(df["origin"]):
        df["origin"] = df["origin"].map(ORIGIN_MAP)
    else:
        df["origin"] = df["origin"].astype(str).str.strip().str.title().replace({"Usa": "USA"})
    if "car_name" in df.columns:
        brand = df["car_name"].astype(str).str.lower().str.split().str[0]
        df["brand"] = brand.replace(BRAND_FIX)
    else:
        df["brand"] = np.nan
    df = df.dropna(subset=["mpg"]).reset_index(drop=True)  # target must be known
    return df


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


def candidate_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Support Vector Regressor": SVR(C=10, epsilon=0.5),
    }


def train_and_select(df: pd.DataFrame):
    """Compare all models with 5-fold CV, pick best by CV RMSE, refit on all data."""
    X, y = df[NUM_COLS + CAT_COLS], df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rows, pipes = [], {}
    for name, est in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        cv = cross_validate(pipe, X_tr, y_tr, cv=5,
                            scoring=("neg_root_mean_squared_error", "r2"))
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": name,
            "CV RMSE": -cv["test_neg_root_mean_squared_error"].mean(),
            "CV R²": cv["test_r2"].mean(),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "Test MAE": mean_absolute_error(y_te, pred),
            "Test R²": r2_score(y_te, pred),
        })
        pipes[name] = pipe
    results = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best = Pipeline([("prep", build_preprocessor()), ("model", candidate_models()[best_name])])
    best.fit(X, y)  # final model uses all data
    joblib.dump({"pipeline": best, "name": best_name, "results": results,
                 "brands": sorted(df["brand"].dropna().unique().tolist()),
                 "ranges": {c: (float(df[c].min()), float(df[c].max()), float(df[c].median()))
                            for c in NUM_COLS}}, MODEL_PATH)
    return best, best_name, results


def predict_mpg(pipeline, specs: dict) -> float:
    row = pd.DataFrame([specs])[NUM_COLS + CAT_COLS]
    return float(pipeline.predict(row)[0])
