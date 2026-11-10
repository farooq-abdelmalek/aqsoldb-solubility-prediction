import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


DESCRIPTOR_FUNCTIONS = {
    "MolLogP": Descriptors.MolLogP,
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": rdMolDescriptors.CalcNumHBD,
    "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
    "NumRotBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "NumArRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumSatRings": rdMolDescriptors.CalcNumSaturatedRings,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "MolMR": Descriptors.MolMR,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "BertzCT": Descriptors.BertzCT,
}


def calculate_descriptors(smiles_list):
    """Calculate 15 RDKit descriptors for each molecule."""

    records = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        row = {}

        for name, function in DESCRIPTOR_FUNCTIONS.items():
            try:
                row[name] = float(function(mol))
            except Exception:
                row[name] = np.nan

        records.append(row)

    return pd.DataFrame(records)

from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Generate binary Morgan fingerprints for a list of SMILES."""

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    fingerprints = np.zeros(
        (len(smiles_list), n_bits),
        dtype=np.uint8
    )

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")

        fingerprint = generator.GetFingerprint(mol)

        DataStructs.ConvertToNumpyArray(
            fingerprint,
            fingerprints[i]
        )

    return fingerprints


def build_feature_matrices(smiles_list):
    """
    Build three molecular feature representations from canonical SMILES.

    Parameters
    ----------
    smiles_list : list of str
        Valid canonical SMILES strings. Their original order is preserved.

    Returns
    -------
    dict
        A dictionary containing three NumPy arrays:

        - "descriptors": 15 physicochemical properties per molecule.
        - "fingerprints": 2048-bit Morgan fingerprints (radius 2).
        - "hybrid": descriptors and fingerprints concatenated together.

    Notes
    -----
    This function performs deterministic molecular feature extraction.

    It does not scale features, impute missing values, or fit any
    preprocessing parameters. These operations must be learned from
    the training set only to avoid data leakage.
    """

    smiles_list = list(smiles_list)

    # Calculate physicochemical descriptors.
    # float32 reduces memory usage during model training.
    descriptors = calculate_descriptors(
        smiles_list
    ).to_numpy(dtype=np.float32)

    # Generate binary Morgan fingerprints.
    fingerprints = calculate_morgan_fingerprints(
        smiles_list,
        radius=2,
        n_bits=2048
    )

    # Combine both representations while preserving molecule order.
    # NumPy converts the binary fingerprints to float32 here.
    hybrid = np.concatenate(
        [descriptors, fingerprints.astype(np.float32)],
        axis=1
    )

    return {
        "descriptors": descriptors,
        "fingerprints": fingerprints,
        "hybrid": hybrid
    }
