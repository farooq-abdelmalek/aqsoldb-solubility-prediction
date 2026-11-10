"""Unit tests for molecular SMILES preprocessing."""

import unittest

from rdkit import Chem

from src.preprocessing import canonicalize_smiles


class TestCanonicalizeSmiles(unittest.TestCase):

    def test_missing_or_empty_smiles(self):
        """Missing and empty inputs should return None."""
        for value in [None, "", "   ", 123]:
            with self.subTest(value=value):
                self.assertIsNone(canonicalize_smiles(value))

    def test_invalid_smiles(self):
        """Malformed molecular structures should be rejected."""
        self.assertIsNone(canonicalize_smiles("C1CC"))

    def test_canonicalization(self):
        """Equivalent SMILES should produce the same representation."""
        self.assertEqual(canonicalize_smiles("OCC"), "CCO")
        self.assertEqual(canonicalize_smiles("CCO"), "CCO")

    def test_salt_components_preserved(self):
        """Canonicalization must not discard disconnected ions."""
        result = canonicalize_smiles("[Na+].[Cl-]")

        self.assertIsNotNone(result)
        self.assertIn("[Na+]", result)
        self.assertIn("[Cl-]", result)

        mol = Chem.MolFromSmiles(result)
        self.assertEqual(len(Chem.GetMolFrags(mol)), 2)

    def test_stereochemistry_preserved(self):
        """Stereochemical information should survive canonicalization."""
        result = canonicalize_smiles("F[C@H](Cl)Br")

        self.assertIsNotNone(result)
        self.assertTrue("@" in result)


if __name__ == "__main__":
    unittest.main()
