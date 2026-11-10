import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

df = pd.read_csv("data/curated-solubility-dataset.csv")

remover = SaltRemover.SaltRemover()
uncharger = rdMolStandardize.Uncharger()

def clean_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    try:
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        mol = uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

df["clean_smiles"] = df["SMILES"].apply(clean_smiles)

valid = df.dropna(subset=["clean_smiles"])
groups = valid.groupby("clean_smiles")["Solubility"]
spreads = groups.max() - groups.min()

print("Original records:", len(df))
print("Invalid after cleaning:", df["clean_smiles"].isna().sum())
print("Unique cleaned structures:", valid["clean_smiles"].nunique())
print("Repeated structure rows:", valid["clean_smiles"].duplicated().sum())
print("Repeated structure groups:", (groups.size() > 1).sum())
print("Groups with LogS spread > 1:", (spreads > 1).sum())

def canonical_only(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

df["canonical_only"] = df["SMILES"].apply(canonical_only)

canonical = df.dropna(subset=["canonical_only"])
canonical_groups = canonical.groupby("canonical_only")["Solubility"]

print("\n--- Canonicalization WITHOUT salt removal ---")
print("Unique canonical structures:", canonical["canonical_only"].nunique())
print("Repeated structure groups:", (canonical_groups.size() > 1).sum())
print("Groups with LogS spread > 1:",
      ((canonical_groups.max() - canonical_groups.min()) > 1).sum())

print("\n--- Largest LogS disagreements ---")

for structure in spreads.sort_values(ascending=False).head(3).index:
    subset = df.loc[
        df["clean_smiles"] == structure,
        ["SMILES", "Solubility"]
    ]

    print("\nCleaned structure:", structure)
    print(subset.to_string(index=False))
    print("LogS spread:", round(spreads[structure], 2))

print("\n--- Invalid and multi-component molecules ---")

print("\nInvalid SMILES:")
print(
    df.loc[
        df["canonical_only"].isna(),
        ["SMILES", "Solubility"]
    ].to_string(index=False)
)

multi_component = canonical["canonical_only"].str.contains(".", regex=False)

print("\nValid molecules:", len(canonical))
print("Multi-component molecules:", multi_component.sum())
print("Single-component molecules:", (~multi_component).sum())
