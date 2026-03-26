import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend - safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

os.makedirs("plots", exist_ok=True)

# 1. load dataset

df = pd.read_csv("insurance.csv")

# Drop row-ID column if present (not a feature)
if "Id" in df.columns:
    df.drop(columns=["Id"], inplace=True)

pd.set_option("display.float_format", "{:.2f}".format)
print("Dataset shape:", df.shape)

# 2. cleaning

print("Duplicate rows:", df.duplicated().sum())
print("Missing values:\n", df.isna().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("Shape after cleaning:", df.shape)
print(df.describe(include="all"))

# 3. EXPLORATORY DATA ANALYSIS

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)

# Numerical feature distributions
numeric_cols = ["age", "bmi", "bloodpressure", "children", "claim"]
df[numeric_cols].hist(bins=20, figsize=(12, 8), color="skyblue", edgecolor="black")
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.savefig("plots/01_numerical_distributions.png", dpi=100)
plt.show()

# Categorical feature distributions
cat_cols = ["gender", "diabetic", "smoker", "region"]
plt.figure(figsize=(12, 8))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("plots/02_categorical_distributions.png", dpi=100)
plt.show()

# Average claim by gender & smoking status
print(df.groupby(["gender", "smoker"])["claim"].mean().round(2))

plt.figure(figsize=(12, 8))
sns.barplot(data=df, x="gender", y="claim", hue="smoker", estimator="mean", errorbar="sd")
plt.title("Average Insurance Claim By Gender & Smoking Status")
plt.tight_layout()
plt.savefig("plots/03_claim_gender_smoker.png", dpi=100)
plt.show()

# Average claim by region & diabetic status
pivot_region_diabetic = df.groupby(["region", "diabetic"])["claim"].mean().unstack()
print(pivot_region_diabetic)

pivot_region_diabetic.plot(kind="bar", figsize=(8, 5))
plt.title("Average Insurance Claim By Region & Diabetic Status")
plt.ylabel("Mean Claim")
plt.tight_layout()
plt.savefig("plots/04_claim_region_diabetic.png", dpi=100)
plt.show()

# Pivot tables
pivot_smoker = pd.pivot_table(df, values="claim", index="region", columns="smoker", aggfunc="mean")
print(pivot_smoker)

pivot_children = pd.pivot_table(df, values="claim", index="children", columns="diabetic", aggfunc="mean")
print(pivot_children)

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/05_correlation_heatmap.png", dpi=100)
plt.show()

# Scatter: Age vs Claim
sns.scatterplot(data=df, x="age", y="claim", hue="smoker", style="gender", alpha=0.7)
plt.title("Claim Vs Age by Smoker & Gender")
plt.tight_layout()
plt.savefig("plots/06_claim_vs_age.png", dpi=100)
plt.show()

# BMI vs Claim
sns.regplot(data=df, x="bmi", y="claim", scatter_kws={"alpha": 0.6})
plt.title("Relationship Between BMI and Claim Amount")
plt.tight_layout()
plt.savefig("plots/07_bmi_vs_claim.png", dpi=100)
plt.show()

# Claim by number of children
sns.boxplot(data=df, x="children", y="claim")
plt.title("Claim Distribution by Number of Children")
plt.tight_layout()
plt.savefig("plots/08_claim_by_children.png", dpi=100)
plt.show()

# 4. FEATURE ENGINEERING

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 30, 45, 60, 100],
    labels=["<18", "18-30", "31-45", "46-60", "60+"]
)
print(df["age_group"].value_counts())

sns.barplot(data=df, x="age_group", y="claim", estimator="mean", errorbar="sd")
plt.title("Average Insurance Claim by Age Group")
plt.tight_layout()
plt.savefig("plots/09_claim_by_age_group.png", dpi=100)
plt.show()

df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)
print(df["bmi_category"].value_counts())

sns.boxplot(data=df, x="bmi_category", y="claim", hue="smoker")
plt.title("Claim Distribution by BMI Category and Smoking Status")
plt.tight_layout()
plt.savefig("plots/10_claim_bmi_smoker.png", dpi=100)
plt.show()

sns.boxplot(data=df, x="bmi_category", y="claim")
plt.title("Claim Distribution by BMI Category")
plt.tight_layout()
plt.savefig("plots/11_claim_by_bmi.png", dpi=100)
plt.show()

# Smoker rate & mean claim by region
region_stats = df.groupby("region").agg(
    smoker_rate=("smoker", lambda x: x.eq("Yes").sum() / x.size * 100),
    mean_claim=("claim", "mean")
).reset_index()
print(region_stats)

fig, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(data=region_stats, x="region", y="smoker_rate", ax=ax1, alpha=0.6)
ax2 = ax1.twinx()
sns.lineplot(data=region_stats, x="region", y="mean_claim", ax=ax2, color="red", marker="o")
ax1.set_ylabel("Smoker Rate (%)")
ax2.set_ylabel("Mean Claim Amount ($)")
plt.title("Smoker Rate and Mean Claim By Region")
plt.tight_layout()
plt.savefig("plots/12_smoker_rate_by_region.png", dpi=100)
plt.show()

# 5. preprocessing

X = df[["age", "gender", "bmi", "bloodpressure", "diabetic", "children", "smoker"]].copy()
y = df["claim"]

encode_cols = ["gender", "diabetic", "smoker"]
label_encoders = {}

for col in encode_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    joblib.dump(le, f"label_encoder_{col}.pkl")
    print(f"Saved encoder: label_encoder_{col}.pkl | Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ["age", "bmi", "bloodpressure", "children"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])
joblib.dump(scaler, "scaler.pkl")
print("Saved scaler.pkl")

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape :", X_test.shape,  y_test.shape)

# 6. model training & evaluation

def evaluate_model(model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    return {
        "R2":   round(r2_score(y_eval, y_pred), 4),
        "MAE":  round(mean_absolute_error(y_eval, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_eval, y_pred)), 2),
    }

results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
results["Linear Regression"] = evaluate_model(lr, X_test, y_test)
print("Linear Regression trained.")

# Polynomial Regression (best degree)
best_poly_score = -np.inf
best_poly_model = None

for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)

    poly_lr = LinearRegression()
    poly_lr.fit(X_train_poly, y_train)
    score = poly_lr.score(X_test_poly, y_test)

    if score > best_poly_score:
        best_poly_score = score
        best_poly_model = (degree, poly, poly_lr)

degree, poly, poly_lr = best_poly_model
results[f"Polynomial Regression (deg={degree})"] = evaluate_model(
    poly_lr, poly.transform(X_test), y_test
)
print(f"Polynomial Regression trained. Best degree: {degree}")

# Random Forest
rf_params = {
    "n_estimators":      [100, 200],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params,
                       cv=3, scoring="r2", n_jobs=-1, verbose=0)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
results["Random Forest"] = evaluate_model(best_rf, X_test, y_test)
print("Random Forest trained. Best params:", rf_grid.best_params_)

# SVR
svr_params = {
    "kernel":  ["linear", "rbf"],
    "C":       [0.1, 10, 50],
    "epsilon": [0.1, 0.2, 0.5],
    "degree":  [2, 3],
}
svr_grid = GridSearchCV(SVR(), svr_params, cv=3, scoring="r2", n_jobs=-1, verbose=0)
svr_grid.fit(X_train, y_train)
best_svr = svr_grid.best_estimator_
results["SVR"] = evaluate_model(best_svr, X_test, y_test)
print("SVR trained. Best params:", svr_grid.best_params_)

# XGBoost
xgb_params = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample":     [0.8, 1.0],
}
xgb_grid = GridSearchCV(XGBRegressor(objective="reg:squarederror", random_state=42),
                        xgb_params, cv=3, scoring="r2", n_jobs=-1, verbose=0)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
results["XGBoost"] = evaluate_model(best_xgb, X_test, y_test)
print("XGBoost trained. Best params:", xgb_grid.best_params_)

# 7. SELECT & SAVE BEST MODEL

results_df = pd.DataFrame(results).T.sort_values(by="R2", ascending=False)
print("\n── Model Comparison ──")
print(results_df.to_string())

models = {
    "Linear Regression":                     lr,
    f"Polynomial Regression (deg={degree})": poly_lr,
    "Random Forest":                         best_rf,
    "SVR":                                   best_svr,
    "XGBoost":                               best_xgb,
}

best_model_name = results_df["R2"].idxmax()
best_model      = models[best_model_name]

joblib.dump(best_model, "best_model.pkl")
print(f"\nBest model: {best_model_name}")
print(f"R²={results_df.loc[best_model_name, 'R2']}  "
      f"MAE={results_df.loc[best_model_name, 'MAE']}  "
      f"RMSE={results_df.loc[best_model_name, 'RMSE']}")
print("Saved best_model.pkl")