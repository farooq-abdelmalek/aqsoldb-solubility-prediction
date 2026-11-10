from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import GroupShuffleSplit


remover = SaltRemover.SaltRemover()
uncharger = rdMolStandardize.Uncharger()


def make_group_key(smiles):
    """Create a grouping key without changing model input features."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = remover.StripMol(mol, dontRemoveEverything=True)
    mol = uncharger.uncharge(mol)

    return Chem.MolToSmiles(mol, canonical=True)


def split_by_group(df, random_state=42):
    """Create reproducible train/validation/test splits."""
    result = df.copy().reset_index(drop=True)

    first = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=random_state
    )

    train_idx, remaining_idx = next(
        first.split(result, groups=result["group_key"])
    )

    remaining = result.iloc[remaining_idx]

    second = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=random_state
    )

    val_relative, test_relative = next(
        second.split(remaining, groups=remaining["group_key"])
    )

    result["split"] = ""
    result.loc[train_idx, "split"] = "train"
    result.loc[remaining_idx[val_relative], "split"] = "validation"
    result.loc[remaining_idx[test_relative], "split"] = "test"

    groups = {
        name: set(result.loc[result["split"] == name, "group_key"])
        for name in ("train", "validation", "test")
    }

    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])

    return result
