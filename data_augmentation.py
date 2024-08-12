# ----------------------------------------------------------------
# TJ Xu et al. Large language model revealing hidden biosynthetic potential of the deep ocean microbiome.
# ----------------------------------------------------------------
# References: 
# https://github.com/huggingface/transformers; 
# https://github.com/DSPsleeporg/smiles-transformer; 
# https://github.com/yydiao1025/Macformer
# https://github.com/Merck/deepbgc
# https://github.com/pschwllr/MolecularTransformer
# https://github.com/gmattedi/SmilesTransformer
import numpy as np
import pandas
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import random

# Function to randomize the core of a molecule
def randomize_core(smiles, core):   
    frag=Chem.MolFromSmiles(core)
    if frag is None:
        return(smiles)
    mol=Chem.MolFromSmiles(smiles)
    mol_atom_list=list(range(mol.GetNumAtoms()))    
    matches=mol.GetSubstructMatches(frag)
    if len(matches)==0:
        return(smiles)
    else:
        match=list(matches[0])
        other_atom_list=[a for a in mol_atom_list if a not in match]
        random.shuffle(other_atom_list)
        new_order=match+other_atom_list
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_order)
        new_mol_smiles=Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False, kekuleSmiles=True)
        return(new_mol_smiles)

# Define constants
enum_factor = 100
max_tries = 500 ## randomized SMILES to generate for each input structure
datatable = "training_data_amplified.csv"
output_path = Path(datatable)
table_columns = ["BGC_features", "SMILES"]
output_path.write_text("\t".join(table_columns) + "\n")
trainingdata_amplified = []
DBfile = "training_data.csv"
df = pd.read_csv(DBfile, sep="\t")
bgc_features, smiles = df['BGC_features'].values, df['SMILES'].values
# Loop through each feature and SMILES pair
for (bgc_feature, smi) in zip(bgc_features, smiles):
    m = Chem.MolFromSmiles(smi)
    core = MurckoScaffold.GetScaffoldForMol(m)
    coresmi = Chem.MolToSmiles(core, isomericSmiles=False, kekuleSmiles=True)
    tries = []
    if len(coresmi) > 0:
        for try_idx in range(max_tries):
            this_try = randomize_core(smi, coresmi)
            tries.append(this_try)
            tries = [rnd for rnd in np.unique(tries)]
            if len(tries) > enum_factor:
                tries = tries[:enum_factor]
                break
    if len(tries) == 0:
        tries.append(smi)
    for sms in tries:
        datapair = {}
        datapair["BGC_features"] = bgc_feature
        datapair["SMILES"] = sms
        trainingdata_amplified.append(datapair)

# Create a dataframe from the amplified training data
res_df = pandas.DataFrame(trainingdata_amplified)
res_df = res_df[table_columns]
train_df = res_df.sample(frac=1)
# Write the dataframe to a CSV file
res_df.to_csv(output_path, sep="\t", header=False, index=False, mode='a')
