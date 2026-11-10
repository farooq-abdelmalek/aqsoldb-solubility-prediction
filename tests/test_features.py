"""Unit tests for molecular feature generation."""

import unittest

import numpy as np

from src.features import (
    DESCRIPTOR_FUNCTIONS,
    calculate_descriptors,
    calculate_morgan_fingerprints,
    build_feature_matrices,
)


class TestMolecularFeatures(unittest.TestCase):

    def setUp(self):
        self.smiles = [
            "CCO",         # Ethanol
            "c1ccccc1",    # Benzene
            "CC(=O)O",     # Acetic acid
        ]

    def test_feature_dimensions(self):
        """All feature representations must have expected dimensions."""
        features = build_feature_matrices(self.smiles)

        self.assertEqual(len(DESCRIPTOR_FUNCTIONS), 15)
        self.assertEqual(features["descriptors"].shape, (3, 15))
        self.assertEqual(features["fingerprints"].shape, (3, 2048))
        self.assertEqual(features["hybrid"].shape, (3, 2063))

    def test_fingerprints_are_binary(self):
        """Morgan fingerprints must contain only zeros and ones."""
        fingerprints = calculate_morgan_fingerprints(self.smiles)

        self.assertTrue(
            np.isin(fingerprints, [0, 1]).all()
        )

        self.assertEqual(fingerprints.dtype, np.uint8)

        # Different molecules should have different fingerprints.
        self.assertFalse(
            np.array_equal(fingerprints[0], fingerprints[1])
        )

    def test_hybrid_feature_consistency(self):
        """Hybrid features must concatenate descriptors and fingerprints."""
        features = build_feature_matrices(self.smiles)

        np.testing.assert_array_equal(
            features["hybrid"][:, :15],
            features["descriptors"],
        )

        np.testing.assert_array_equal(
            features["hybrid"][:, 15:],
            features["fingerprints"],
        )

    def test_determinism_and_molecule_order(self):
        """Repeated calculations must preserve values and input order."""
        first = build_feature_matrices(self.smiles)
        second = build_feature_matrices(self.smiles)

        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

        descriptors = calculate_descriptors(self.smiles)

        # Ethanol and benzene have different molecular weights.
        self.assertAlmostEqual(
            descriptors.loc[0, "MolWt"], 46.069, places=2
        )

        self.assertAlmostEqual(
            descriptors.loc[1, "MolWt"], 78.114, places=2
        )

    def test_invalid_smiles_rejected(self):
        """Feature extraction should reject invalid molecular structures."""
        with self.assertRaises(ValueError):
            calculate_descriptors(["C1CC"])

        with self.assertRaises(ValueError):
            calculate_morgan_fingerprints(["C1CC"])

        with self.assertRaises(ValueError):
            build_feature_matrices(["CCO", "C1CC"])


if __name__ == "__main__":
    unittest.main()
