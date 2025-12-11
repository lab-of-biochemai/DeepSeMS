import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from sascorer.sascorer import calculateScore as sascore
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def calculate_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'Molecular Formula': None,
            'Molecular Weight': None,
            'QED': None,
            'SAScore': None
        }
    properties = {
        'Molecular Formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
        'Molecular Weight': Descriptors.MolWt(mol),
        'QED': QED.qed(mol),
        'SAScore': sascore(mol)
    }
    return properties

def process_csv(input_file, output_dir):

    if output_dir is None:
        output_dir = os.path.dirname(input_file)

    df = pd.read_csv(input_file, sep="\t")

    properties_list = []
    for smiles in df['Predicted SMILES']:
        properties = calculate_molecular_properties(smiles)
        print(f"Predicted SMILES: {smiles}")
        print(f"Molecular Formula: {properties['Molecular Formula']}")
        print(f"Molecular Weight: {properties['Molecular Weight']:.2f}")
        print(f"QED: {properties['QED']:.2f}")
        print(f"SAScore: {properties['SAScore']:.2f}")
        properties_list.append(properties)
    
    properties_df = pd.DataFrame(properties_list)
    
    result_df = pd.concat([df, properties_df], axis=1)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_molecular_properties.csv")
    
    result_df.to_csv(output_file, sep="\t", header=True, index=False, mode='w')
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate molecular properties for SMILES strings in a CSV file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, required=False, default=None, 
                        help="Directory to save the output CSV file. Defaults to the directory of the input file.")
    args = parser.parse_args()
    process_csv(args.input_file, args.output_dir)