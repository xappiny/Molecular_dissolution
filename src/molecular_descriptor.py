import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, SaltRemover

# 读取Excel文件
input_file = r"\smiles.csv"  
output_file = r"\descriptor.csv" 

data = pd.read_csv(input_file, encoding='utf-8-sig')

smiles_columns = {
    "API_smiles": "_API",
    "Polymer_smiles": "_excip1",
    "Surfant_smiles": "_excip2"
}

descriptor_names = [desc[0] for desc in Descriptors.descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

remover = SaltRemover.SaltRemover()

def remove_salt(smiles):
    if pd.isna(smiles):
        return None

    smiles_str = str(smiles).strip()

    if smiles_str in ["", "0"]:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            return Chem.MolToSmiles(remover.StripMol(mol))
    except Exception as e:
        print(f"Error processing SMILES: {smiles_str}, Error: {e}")
    return None


def calculate_descriptors(smiles):
    if pd.isna(smiles):
        return [0] * len(descriptor_names)

    smiles_str = str(smiles).strip()

    if smiles_str in ["", "0"]:
        return [0] * len(descriptor_names)

    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            return calculator.CalcDescriptors(mol)
    except Exception as e:
        print(f"Error calculating descriptors for SMILES: {smiles_str}, Error: {e}")
    return [0] * len(descriptor_names)


for col, suffix in smiles_columns.items():
    if col in data.columns:
        data[col] = data[col].astype(str)
        descriptor_results = data[col].apply(calculate_descriptors).tolist()
        descriptor_df = pd.DataFrame(descriptor_results, columns=[f"{name}{suffix}" for name in descriptor_names])
        data = pd.concat([data, descriptor_df], axis=1)

data.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"save to {output_file}")
