# AqSolDB — Aqueous Solubility Prediction

Reproducible molecular machine-learning pipeline for predicting aqueous solubility (LogS) from SMILES, using RDKit, scikit-learn, XGBoost, LightGBM, Optuna, MLflow, Docker, and Nextflow DSL2.

## Results

**Final selected model:** Optuna-tuned XGBoost with hybrid RDKit descriptors and Morgan fingerprints.

| Metric | Held-out test |
|---|---:|
| R² | **0.8044** |
| RMSE | **1.0554** |
| MAE | **0.6948** |

The final model was trained using 8,967 development compounds and evaluated once on 1,013 held-out compounds. RMSE and MAE are reported in LogS units, where LogS = log10(solubility in mol/L).

The model-selection cross-validation R² was **0.8059**; this is not the held-out test result.

## Dataset

The project uses **AqSolDB**, a curated collection of experimental aqueous-solubility measurements published by Sorkun, Khetan, and Er (2019).

- Source: [Harvard Dataverse — AqSolDB](https://doi.org/10.7910/DVN/OVHAW8)
- Publication: [Scientific Data (2019)](https://doi.org/10.1038/s41597-019-0151-1)
- Original dataset license: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- Original dataset size: 9,982 records
- Valid SMILES retained for modeling: 9,980 records

The dataset is stored at `data/curated-solubility-dataset.csv`. The frozen dataset checksum is recorded in `results/selected_model.json`.

## Methodology

### 1. Molecular preprocessing

SMILES strings are parsed and canonicalized using RDKit, preserving stereochemistry and disconnected chemical components.

Two invalid SMILES records are excluded. The feature-generation representation preserves salts; a separate, salt-insensitive and charge-normalized molecular identity is used for grouping.

### 2. Leakage-aware data splitting

Two successive `GroupShuffleSplit` operations partition molecular groups into approximately 80% training, 10% validation, and 10% test.

Related representations assigned the same normalized molecular group cannot cross split boundaries.

This is **molecular-identity grouping, not Bemis–Murcko scaffold splitting**. Therefore, the held-out results should not be interpreted as evidence of scaffold-level generalization.

The held-out test set remains excluded from model selection and hyperparameter optimization.

### 3. Molecular representations

Three feature representations are compared:

| Representation | Dimensions |
|---|---:|
| RDKit physicochemical descriptors | 15 |
| Binary Morgan fingerprints (radius 2) | 2,048 |
| Hybrid descriptors + fingerprints | 2,063 |

Missing-value imputation and, where needed, feature scaling are handled inside scikit-learn pipelines rather than fitted globally before evaluation.

### 4. Model comparison and selection

The initial study compares five model families across the three feature representations:

- Ridge Regression
- Support Vector Regression
- Random Forest
- XGBoost
- LightGBM

Fifteen baseline combinations are compared using group-aware three-fold cross-validation.

Selected baseline CV results:

| Model | Representation | Mean CV R² |
|---|---|---:|
| LightGBM | Hybrid | 0.8033 |
| XGBoost | Hybrid | 0.8014 |
| XGBoost | Descriptors | 0.7639 |
| Random Forest | Descriptors | 0.7627 |

The two leading model families are subsequently tuned with Optuna. The frozen selected XGBoost configuration achieves **0.8059 mean CV R²**.

This does not establish statistically significant superiority over LightGBM. Model selection and the final test result are documented separately.

## Reproduce training and inference

### Requirements

- Docker
- Nextflow DSL2
- Java compatible with your installed Nextflow version
- Linux or WSL2 recommended

For Windows users, run Nextflow inside WSL2 with Docker Desktop's WSL integration enabled.

### Quick start

From a Linux/WSL terminal in the project root:

```bash
docker build -t aqsoldb-solubility:dev .

printf 'SMILES\nCCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O\n' > molecules.csv

nextflow run . --input_csv molecules.csv
```

The default Nextflow workflow:

1. Loads the versioned dataset and frozen model-selection configuration.
2. Reconstructs the selected XGBoost model using development data only.
3. Generates predictions for the supplied SMILES.
4. Publishes `pipeline_outputs/nextflow/predictions.csv`.

To reuse cached Nextflow tasks:

```bash
nextflow run . --input_csv molecules.csv -resume
```

To skip training and use an existing trusted model:

```bash
nextflow run . \
  --input_csv molecules.csv \
  --model_path artifacts/final_model.joblib
```

The model file is generated locally and is not committed to Git. The workflow does not rerun hyperparameter tuning or the held-out test evaluation.

## Automated tests

Four modules provide 20 unit tests covering:

- SMILES canonicalization and validation
- Molecular-group splitting and reproducibility
- Descriptor and fingerprint generation
- Prediction-interface behavior and error handling

Build the image, then run the tests from Linux/WSL:

```bash
docker run --rm \
  --mount "type=bind,source=$(pwd)/tests,target=/app/tests,readonly" \
  aqsoldb-solubility:dev \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

GitHub Actions is configured in `.github/workflows/tests.yml` to run the unit tests on pushes and pull requests. Its hosted execution must be verified after GitHub publication.

## Repository structure

```text
.github/workflows/  GitHub Actions
data/               AqSolDB dataset
notebooks/          Exploratory notebook
results/            Recorded model selection and evaluation results
scripts/            Benchmarking, tuning, training, and inference
src/                Molecular preprocessing, splitting, and features
tests/              Automated unit tests
Dockerfile          Reproducible container environment
main.nf             Nextflow training and inference workflow
nextflow.config     Nextflow execution configuration
requirements.txt    Python dependencies
```

## Scientific limitations

- **Generalization:** Group-aware random splitting does not test extrapolation to unseen molecular scaffolds or an independent external dataset.
- **Dataset uncertainty:** AqSolDB combines experimental measurements from multiple sources with varying reliability.
- **Chemical applicability:** Predictions for molecules outside the training data's chemical domain may be unreliable.
- **Reproducibility:** The dataset checksum and frozen model parameters are recorded, but direct Python dependencies are pinned, but transitive dependencies are not fully locked.
- **Intended use:** The model is a research demonstration, not a validated regulatory, clinical, or pharmaceutical decision-making tool.

## Research records

The repository retains:

- `results/cv_summary.csv` — Baseline cross-validation comparison
- `results/cv_fold_scores.csv` — Fold-level baseline results
- `results/selected_model.json` — Frozen selection and model hyperparameters
- `results/final_test_metrics.json` — Final held-out evaluation
- `results/final_test_predictions.csv` — Recorded held-out predictions

The final test set was evaluated once during the original modeling workflow. Reproduction of inference does not require evaluating it again.

## Citation and data attribution

Sorkun, M. C., Khetan, A., & Er, S. (2019). *AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds*. Scientific Data, 6, 143.

https://doi.org/10.1038/s41597-019-0151-1

Original dataset: https://doi.org/10.7910/DVN/OVHAW8

Dataset license: CC0 1.0. Original project source code is licensed under MIT (see LICENSE).
