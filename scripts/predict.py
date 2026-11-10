"""
Predict aqueous solubility (LogS) from molecular SMILES.

Examples
--------
Predict individual molecules:
    python -m scripts.predict --smiles "CCO" "c1ccccc1"

Predict molecules from a CSV file:
    python -m scripts.predict --input molecules.csv --output predictions.csv

The fitted model is loaded from disk.
No model training or test-set evaluation takes place.

Only load model files produced by this trusted project.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import canonicalize_smiles
from src.features import build_feature_matrices


DEFAULT_MODEL = Path("artifacts/final_model.joblib")


def load_model(model_path):
    """Load and validate the saved model bundle."""

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    bundle = joblib.load(model_path)

    if (
        bundle["feature_set"] != "hybrid"
        or bundle["n_features"] != 2063
    ):
        raise ValueError(
            "Unexpected model feature configuration."
        )

    return bundle["pipeline"]


def predict_smiles(smiles_list, model):
    """
    Canonicalize SMILES, calculate features, and predict LogS.

    Invalid SMILES raise an error instead of being silently
    discarded, preserving the input-to-prediction mapping.
    """

    canonical = []

    for index, smiles in enumerate(smiles_list, start=1):

        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(
                f"Missing SMILES at input row {index}."
            )

        molecule = canonicalize_smiles(smiles)

        if molecule is None:
            raise ValueError(
                f"Invalid SMILES at input row {index}: {smiles}"
            )

        canonical.append(molecule)

    # Use the same feature-generation implementation as training.
    X = build_feature_matrices(canonical)["hybrid"]

    if X.shape[1] != 2063:
        raise ValueError("Unexpected feature matrix shape.")

    predictions = model.predict(X)

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    return pd.DataFrame({
        "input_smiles": smiles_list,
        "canonical_smiles": canonical,
        "predicted_logS": predictions,
    })


def main():
    """Parse command-line arguments and generate predictions."""

    parser = argparse.ArgumentParser(
        description="Predict molecular aqueous solubility (LogS)."
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--smiles",
        nargs="+",
        help="One or more molecular SMILES strings.",
    )

    source.add_argument(
        "--input",
        type=Path,
        help="Input CSV file containing molecular SMILES.",
    )

    parser.add_argument(
        "--smiles-column",
        default="SMILES",
        help="SMILES column name in the input CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV file.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the saved model bundle.",
    )

    args = parser.parse_args()

    if args.input is not None:
        df = pd.read_csv(args.input)

        if args.smiles_column not in df.columns:
            parser.error(
                f"Column '{args.smiles_column}' not found."
            )

        smiles_list = df[args.smiles_column].tolist()

    else:
        smiles_list = args.smiles

    if not smiles_list:
        parser.error("No molecules were provided.")

    model = load_model(args.model)
    results = predict_smiles(smiles_list, model)

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(args.output, index=False)

        print(
            f"Saved {len(results)} predictions to {args.output}"
        )

    else:
        print(results.to_string(index=False))

    print("PASS: Molecular inference completed!")


if __name__ == "__main__":
    main()
