"""Unit tests for the molecular prediction interface."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.predict import load_model, predict_smiles


class DummyModel:
    """Minimal model for testing the prediction interface."""

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)
        self.input_shape = None

    def predict(self, X):
        self.input_shape = X.shape
        return self.predictions


class TestMolecularPrediction(unittest.TestCase):

    def test_predictions_preserve_order(self):
        """Predictions must correspond to the original SMILES order."""
        model = DummyModel([1.25, -2.0])

        result = predict_smiles(
            ["OCC", "c1ccccc1"],
            model,
        )

        self.assertEqual(model.input_shape, (2, 2063))

        self.assertEqual(
            result["input_smiles"].tolist(),
            ["OCC", "c1ccccc1"],
        )

        self.assertEqual(
            result["canonical_smiles"].tolist(),
            ["CCO", "c1ccccc1"],
        )

        np.testing.assert_allclose(
            result["predicted_logS"].to_numpy(),
            [1.25, -2.0],
        )

    def test_missing_smiles_rejected(self):
        """Empty and missing SMILES must raise an informative error."""
        for invalid in ["", "  ", None]:
            with self.subTest(invalid=invalid):
                model = DummyModel([1.0, 2.0])

                with self.assertRaisesRegex(
                    ValueError, "input row 2"
                ):
                    predict_smiles(["CCO", invalid], model)

    def test_invalid_smiles_rejected(self):
        """Malformed SMILES must not reach model prediction."""
        model = DummyModel([1.0, 2.0])

        with self.assertRaisesRegex(
            ValueError, "Invalid SMILES at input row 2"
        ):
            predict_smiles(["CCO", "C1CC"], model)

        self.assertIsNone(model.input_shape)

    def test_nonfinite_predictions_rejected(self):
        """NaN and infinite predictions must raise errors."""
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                model = DummyModel([value])

                with self.assertRaisesRegex(
                    ValueError, "non-finite"
                ):
                    predict_smiles(["CCO"], model)

    def test_model_bundle_validation(self):
        """Only the expected feature configuration is accepted."""
        model = DummyModel([1.0])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_model.joblib"
            path.touch()

            valid_bundle = {
                "pipeline": model,
                "feature_set": "hybrid",
                "n_features": 2063,
            }

            with patch(
                "scripts.predict.joblib.load",
                return_value=valid_bundle,
            ):
                self.assertIs(load_model(path), model)

            for field, invalid_value in [
                ("feature_set", "descriptors"),
                ("n_features", 15),
            ]:
                bundle = valid_bundle.copy()
                bundle[field] = invalid_value

                with patch(
                    "scripts.predict.joblib.load",
                    return_value=bundle,
                ):
                    with self.assertRaises(ValueError):
                        load_model(path)

            with self.assertRaises(FileNotFoundError):
                load_model(Path(directory) / "missing.joblib")


if __name__ == "__main__":
    unittest.main()
