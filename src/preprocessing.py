from rdkit import Chem


def canonicalize_smiles(smiles):
    """Canonicalize a SMILES without removing chemical components."""

    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )
