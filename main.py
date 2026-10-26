
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# 1. Ucitavanje podataka
df = pd.read_csv("cars.csv")
df.columns = df.columns.str.strip()

print("Pocetni oblik podataka:", df.shape)
print("\nNedostajuce vrijednosti:")
print(df.isnull().sum())

# 2. Ciscenje podataka
df = df.drop_duplicates()

df["priceUSD"] = pd.to_numeric(df["priceUSD"], errors="coerce")
df = df.dropna(subset=["priceUSD"])
df = df[df["priceUSD"] > 0]

# Pretvaranje poznatih numerickih kolona u brojeve
for column in ["year", "mileage(kilometers)", "volume(cm3)"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# 3. Inzenjering karakteristika
if "year" in df.columns:
    df["car_age"] = (2026 - df["year"]).clip(lower=0, upper=100)

if "mileage(kilometers)" in df.columns and "car_age" in df.columns:
    df["mileage_per_year"] = (
        df["mileage(kilometers)"] / df["car_age"].clip(lower=1)
    )

if "volume(cm3)" in df.columns:
    df["engine_volume_liters"] = df["volume(cm3)"] / 1000

print("\nOblik podataka nakon ciscenja:", df.shape)

# 4. Ulazni podaci i ciljna cijena
X = df.drop(columns=["priceUSD"])
y = df["priceUSD"]

# Uklanjanje kolona koje su potpuno prazne
X = X.dropna(axis=1, how="all")

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

# 5. Pretprocesiranje
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# 6. Podjela podataka
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Tri modela, uz ogranicenu velicinu stabala
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        min_samples_leaf=10,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_mae = float("inf")

# 8. Treniranje i evaluacija
for name, algorithm in models.items():
    print(f"\nTreniranje modela: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE: ${mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2: {r2:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_model = pipeline

# 9. Poredjenje modela
results_df = pd.DataFrame(results).sort_values("MAE")

print("\nPOREDJENJE MODELA:")
print(results_df.to_string(index=False))

print("\nNajbolji model:", results_df.iloc[0]["Model"])

# 10. Primjeri predvidjanja
predictions = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Stvarna cijena": y_test.to_numpy()[:10],
    "Predvidjena cijena": predictions[:10]
})

comparison["Greska"] = abs(
    comparison["Stvarna cijena"] -
    comparison["Predvidjena cijena"]
)

print("\nPRIMJERI PREDVIDJANJA:")
print(comparison.round(2).to_string(index=False))

# 11. Cuvanje najboljeg modela
joblib.dump(
    best_model,
    "car_price_model.joblib",
    compress=3
)

print("\nModel je sacuvan u car_price_model.joblib")

import os
size_mb = os.path.getsize("car_price_model.joblib") / (1024 * 1024)
print(f"Velicina modela: {size_mb:.2f} MB")

if size_mb >= 25:
    print("Model je i dalje prevelik za direktan GitHub upload.")
else:
    print("Model je manji od 25 MB i mozes pokusati upload.")
