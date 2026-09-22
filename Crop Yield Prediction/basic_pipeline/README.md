# Run the Crop Yield Prediction benchmark

The historical-data CLI is runnable. Its previously missing package files have
been reimplemented for this workflow; this is a new baseline implementation,
not a recovery of the original model. The Alberta satellite workflows remain
incomplete: several scripts, helper functions, satellite exports, and observed
field-yield inputs referenced in the older handoff documentation are absent.

## 1. Open the project directory

```bash
cd "/Users/meeko/Work/crop-prediction/Crop Yield Prediction/basic_pipeline"
```

Run the following commands from this directory. Quotes protect the spaces in
the path. This is a command-line project; it does not start a website.

## 2. Create and activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The first command creates an isolated environment for the project's packages.
The second makes `python`, `pip`, and installed commands use that environment
in the current terminal. Setup has already been completed on this machine;
in a new terminal, you only need to activate it.

## 3. Install the package and modeling dependencies

```bash
python -m pip install -e '.[ml]'
```

`-e` installs this source directory in editable mode, so source edits take effect
without reinstalling. `[ml]` adds NumPy, pandas, and scikit-learn. Installation
also registers the `yield-prediction` terminal command. Internet access is needed
to download dependencies; the actual historical benchmark runs locally.

The larger `requirements.txt` includes geospatial and Earth Engine tools used by
other workflows. Those are not needed for this benchmark. For a summary alone,
`python -m pip install -e .` suffices.

## 4. Validate and summarize the data

```bash
yield-prediction --data archive/yield_df.csv summary
```

Reads the CSV, ignores its saved index column, checks required fields and numeric
values, and reports its coverage. It does not change the original file or train
a model. Expected output:

```text
rows: 28242
areas: 101
items: 10
year_min: 1990
year_max: 2013
duplicate_records: 2310
```

## 5. Train and evaluate

```bash
yield-prediction --data archive/yield_df.csv benchmark
```

This command:

1. Removes 2,310 exact duplicate records, leaving 25,932 records.
2. Uses area, crop, year, rainfall, pesticide use, and temperature as predictors.
   The target is `hg/ha_yield`, measured in hectograms per hectare.
3. Reserves the newest 20% of distinct years for evaluation: 1990–2008 is the
   training period and 2009–2013 is the held-out test period for this dataset.
4. Encodes crop and area categories using only the training data. Unseen test
   categories are allowed without fitting the encoder on test data.
5. Fits a mean-yield baseline and a 100-tree random forest. The forest uses a
   fixed random seed for repeatability within the same software environment.
6. Predicts the held-out records and compares predictions with observed yields.
7. Writes `data/processed/benchmark/metrics.csv` and `predictions.csv`.

MAE is the average absolute prediction error; RMSE penalizes large errors more
strongly. Lower values are better for both. R² measures fit relative to the test
set mean: 1 is perfect, 0 matches that mean, and negative values are worse.
Divide errors or predictions in hg/ha by 10,000 to convert to tonnes/hectare.

This evaluates historical records with their recorded predictors. It does not
establish pre-season forecast performance, estimate Alberta pixel yields, or
save a deployable fitted model. Pooled metrics also combine crops with very
different yield scales. Rerunning replaces the two result files; choose a new
directory to keep a separate run:

```bash
yield-prediction --data archive/yield_df.csv benchmark --output-dir data/processed/my_run
```

## 6. Run checks or use the combined launcher

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
bash scripts/run_basic_pipeline.sh
```

The tests check dataset validation, CLI behavior, separation of training and
test years, duplicate removal, and prediction output. The launcher runs the
summary and then the benchmark when its dependencies are available.

Use `deactivate` when finished to leave the Python environment.

## Alberta workflow status

The historical CSV does not contain Alberta field-level satellite features.
The older guides describe extraction, preprocessing, and pixel modeling, but
required files are missing, including `alberta_ats.py`, `gee/`, the full-field
preprocessor, batch utility modules, `reference_inputs/`, and pixel training
functions in `modeling.py`. These workflows need restoration and their real
input data before they can run. The surviving NDVI redistribution code also
requires known field yield and a prepared pixel table; it does not independently
predict an unknown field's yield.
