"""Unit tests for leakage-safe molecular dataset splitting."""

import unittest

import pandas as pd

from src.splitting import make_group_key, split_by_group


class TestMolecularSplitting(unittest.TestCase):

    def setUp(self):
        """Create 30 molecular groups with three rows each."""
        self.df = pd.DataFrame({
            "group_key": [
                f"molecule_{i}"
                for i in range(30)
                for _ in range(3)
            ],
            "value": range(90),
        })

    def test_invalid_smiles(self):
        """Invalid molecular structures should not receive a key."""
        self.assertIsNone(make_group_key("C1CC"))

    def test_equivalent_smiles(self):
        """Different SMILES for the same molecule must share a key."""
        self.assertEqual(
            make_group_key("CCO"),
            make_group_key("OCC"),
        )

    def test_salt_and_charge_normalization(self):
        """Counterions and charge states should not split molecular groups."""
        self.assertEqual(
            make_group_key("CC(=O)[O-].[Na+]"),
            make_group_key("CC(=O)O"),
        )

    def test_no_group_leakage(self):
        """A molecular group must belong to exactly one split."""
        result = split_by_group(self.df, random_state=42)

        self.assertEqual(len(result), len(self.df))

        self.assertEqual(
            set(result["split"]),
            {"train", "validation", "test"},
        )

        groups = {
            split: set(result.loc[
                result["split"] == split, "group_key"
            ])
            for split in ("train", "validation", "test")
        }

        self.assertTrue(
            groups["train"].isdisjoint(groups["validation"])
        )
        self.assertTrue(
            groups["train"].isdisjoint(groups["test"])
        )
        self.assertTrue(
            groups["validation"].isdisjoint(groups["test"])
        )

        # Every group must be assigned to exactly one subset.
        self.assertTrue(
            (result.groupby("group_key")["split"].nunique() == 1).all()
        )

    def test_reproducibility_and_no_mutation(self):
        """The same random seed must reproduce the same assignments."""
        original = self.df.copy(deep=True)

        first = split_by_group(self.df, random_state=42)
        second = split_by_group(self.df, random_state=42)

        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(self.df, original)


if __name__ == "__main__":
    unittest.main()
