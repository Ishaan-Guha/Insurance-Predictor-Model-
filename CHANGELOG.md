# Changelog

All notable changes to this project are documented here.

---

## [1.1.0]

### Fixed — `train_model.py`

- **`sns.set()` deprecated** — replaced with `sns.set_theme()` as required by seaborn's current API. The old call still works but raises a deprecation warning on every run.
- **`bool.mean()` unresolved attribute** — the lambda `(x == "Yes").mean() * 100` was flagged because `==` can return a plain `bool`, making `.mean()` unresolvable to the type checker. Replaced with `x.eq("Yes").sum() / x.size * 100` which operates entirely on `pd.Series` methods and resolves cleanly.
- **`evaluate_model` parameter shadowing** — function parameters were named `X_test` and `y_test`, identical to the outer-scope train/test split variables. Renamed to `X_eval` and `y_eval` to remove the scope conflict.
- **`plt.title = (...)` silent no-op** — all chart titles were being assigned as attributes instead of called as functions (e.g. `plt.title = "..."` instead of `plt.title("...")`). None of the titles were rendering. Fixed across all 12 plots.
- **Colab-specific code removed** — `from google.colab import drive` and `drive.mount(...)` removed. These crash the script outside of Colab.
- **Hardcoded `/content/insurance.csv` path** — changed to relative `insurance.csv` so the script works locally without modification.
- **No `random_state` on train/test split** — added `random_state=42` for reproducible results across runs.
- **`Id` column included in features** — the dataset contains a row-ID column that is not a feature. Added a drop at load time if the column is present.
- **`bool` unresolved `astype`** — an intermediate fix using `.astype(int).mean()` was itself flagged by PyCharm since `.astype()` is not recognised on `bool`. Resolved fully by switching to the `.eq().sum() / .size` approach described above.

### Fixed — `app.py`

- **Unused `import numpy as np`** — numpy was imported but never called in the app. Removed.
- **`load_artifacts()` shadowing outer-scope names** — variables inside the cached function (`scaler`, `le_gender`, `le_diabetic`, `le_smoker`, `model`) had the same names as the module-level variables they were assigned to, triggering "shadows name from outer scope" on every line. Prefixed all internal names with `_` to make scope explicit.

### Added

- **PyCharm run configuration instructions** in README — documents the correct way to launch a Streamlit app from PyCharm (via `streamlit.exe run app.py`) and explains why running `app.py` directly with the Python interpreter produces `missing ScriptRunContext` warnings without opening a browser.
- **`plots/` directory** — `train_model.py` now creates this folder automatically and saves all 12 EDA charts there on each training run.
- **Missing artifact check in `app.py`** — app shows a clear error message pointing to `train_model.py` if any `.pkl` file is absent, rather than crashing with a `FileNotFoundError`.
- **`@st.cache_resource` on artifact loading** — prevents the model and encoders from being reloaded from disk on every user interaction.
- **Risk level indicator** — prediction result now includes a contextual Low / Moderate / High risk note based on the predicted claim value.

### Structure

- Separated the original single Colab notebook into two standalone files: `train_model.py` (EDA + training) and `app.py` (Streamlit UI).
- Added `requirements.txt` with pinned minimum versions for all dependencies.

---

## [1.0.0] - Initial Release

- Original Google Colab notebook combining EDA, model training, and Streamlit UI in a single file.
- Five models trained: Linear Regression, Polynomial Regression, Random Forest, SVR, XGBoost.
- Best model selected automatically by R² score and saved via joblib.
