# Red Wine Quality Prediction – Regression Project

A modular ML project that predicts the quality score (3–8) of red wines from their chemical properties. Each stage is its own Python module, so it can be wired into a DVC pipeline.

```
data_ingestion → data_transformation → model_trainer → model_evaluation
```

## Dataset

[UCI Wine Quality – Red](https://archive.ics.uci.edu/dataset/186/wine+quality): 1,599 Portuguese "Vinho Verde" red wines. It is downloaded automatically by the ingestion stage.

| Features (11) | Target |
|---|---|
| fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol | **quality**: taster score, 3 (worst) to 8 (best) |

Most wines score 5 or 6; very few score 3 or 8.

## Project structure

```
MLOPS/
├── src/
│   ├── config.py               # File paths and column names
│   ├── logger.py               # Logging to console + logs/pipeline.log
│   ├── utils.py                # Load params, save/load pickle and JSON
│   ├── mlflow_utils.py         # MLflow setup + passing the run id between stages
│   ├── data_ingestion.py       # Stage 1: download the dataset
│   ├── data_transformation.py  # Stage 2: clean, add features, split, build preprocessor
│   ├── model_trainer.py        # Stage 3: compare 4 models, save the best
│   ├── model_evaluation.py     # Stage 4: test-set metrics
│   └── predict.py              # Predict quality for new wines
├── main.py                     # Runs all 4 stages
├── params.yaml                 # Settings (data URL, test size, model hyperparameters)
└── requirements.txt
```

## What each stage does

1. **data_ingestion**: downloads the CSV and renames columns to snake_case → `data/raw/winequality_red.csv`
2. **data_transformation**:
   - removes 240 duplicate rows
   - adds 3 features: `total_acidity`, `free_so2_ratio`, `alcohol_x_sulphates`
   - splits 80/20 into train and test, stratified by quality so rare scores appear in both → `data/processed/`
   - builds a preprocessor (fills missing values with the median, then scales), fitted on train only → `models/preprocessor.pkl`
3. **model_trainer**: compares Linear Regression, Ridge, Random Forest and Gradient Boosting with 5-fold cross-validation, then retrains the best one on all training data → `models/model.pkl`, `reports/training_report.json`. Logs everything to MLflow.
4. **model_evaluation**: scores the model on the test set → `reports/metrics.json`. Logs the test metrics to the same MLflow run, then logs and registers the model.

## DagsHub setup

[DagsHub](https://dagshub.com) gives every repo a hosted **MLflow tracking server** and a **DVC remote**.

1. Create a repo on DagsHub (or connect your GitHub repo).
2. Get a token: your avatar → **Settings → Tokens**.
3. Copy `.env.example` to `.env` and fill it in:
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
   MLFLOW_TRACKING_USERNAME=<username>
   MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
   ```
   `.env` is git-ignored, so your token is never committed. The code loads it automatically.
4. Run `python main.py`. Runs appear in your DagsHub repo under **Experiments**; the MLflow UI is at `https://dagshub.com/<username>/<repo>.mlflow`.

To track locally instead, set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` in `.env` and view it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### DagsHub as DVC remote (optional)

Stores your data and model files on DagsHub. Run once after `dvc init`:

```powershell
dvc remote add -d origin https://dagshub.com/<username>/<repo>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <your-dagshub-token>
```

`--local` saves the credentials in `.dvc/config.local`, which is not committed. Then use `dvc push` / `dvc pull`. Your DagsHub repo's **Remote** button shows these commands with your details filled in.

## MLflow tracking

Settings are in `params.yaml` under `mlflow`:

```yaml
mlflow:
  tracking_uri: https://dagshub.com/<username>/<repo>.mlflow   # overridden by MLFLOW_TRACKING_URI in .env
  experiment_name: red-wine-quality
  registered_model_name: red-wine-quality-model
```

Every pipeline run creates one parent run, `training`, with a child run for each candidate model:

```
training (parent)          params: cv_folds, test_size, n_train_rows ...
│                          metrics: best_cv_r2, test_r2, test_mae, test_rmse ...
│                          tag: best_model
│                          artifacts: training_report.json, metrics.json, model/
├── linear_regression      params + cv_r2_mean, cv_r2_std
├── ridge                  params + cv_r2_mean, cv_r2_std
├── random_forest          params + cv_r2_mean, cv_r2_std
└── gradient_boosting      params + cv_r2_mean, cv_r2_std
```

The logged model is the preprocessor and model together, so it takes the 14 feature columns as input. Each run adds a new version to the model registry (version 1, 2, 3 …).

Training and evaluation run as separate processes. The trainer writes its run id to `reports/mlflow_run.json`, and evaluation uses it to log into the same run.

### Open the MLflow UI

With DagsHub, open `https://dagshub.com/<username>/<repo>.mlflow`. If you're tracking locally, run `mlflow ui --backend-store-uri sqlite:///mlflow.db` and open http://127.0.0.1:5000. In the UI you can:
- **Experiments → red-wine-quality**: see every run; expand a parent run to see its 4 models
- Tick several runs and click **Compare** to chart metrics against params
- **Models → red-wine-quality-model**: see all registered versions

### Load a registered model

```python
import mlflow
from dotenv import load_dotenv
load_dotenv()   # picks up MLFLOW_TRACKING_URI and credentials from .env
model = mlflow.pyfunc.load_model("models:/red-wine-quality-model/latest")
model.predict(df[FEATURES])   # df must already have the engineered features (see add_features)
```

### Try it

Change a hyperparameter in `params.yaml` (e.g. `random_forest.max_depth: 8`), run `python main.py` again, then compare the two runs in the UI.

## Setup

```powershell
python -m venv mlops_env
.\mlops_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If PowerShell says "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and activate again.

## Run

All stages:
```powershell
python main.py
```

One stage at a time (run from the project root, with `-m`):
```powershell
python -m src.data_ingestion
python -m src.data_transformation
python -m src.model_trainer
python -m src.model_evaluation
```

Predict for new wines:
```powershell
python -m src.predict
```

## Metrics

| Metric | Meaning |
|---|---|
| r2 | Share of quality variation the model explains (1.0 = perfect) |
| mae | Average error in quality points |
| rmse | Like MAE but punishes big errors more |
| rounded_accuracy | How often the rounded prediction equals the true score exactly |
| within_1_point | How often the rounded prediction is within 1 point of the true score |

Expected results: R² ≈ 0.40, MAE ≈ 0.49, about 60% exact and about 97% within 1 point. Wine quality is a subjective taster score, so R² around 0.4 is normal for this dataset.

## Changing settings

Edit `params.yaml`, for example:
- `data_transformation.test_size`: test split
- `model_trainer.models`: hyperparameters, or remove a model from the comparison
- `model_trainer.cv_folds`: number of cross-validation folds
