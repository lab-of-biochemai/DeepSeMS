# ----------------------------------------------------------------
# TJ Xu et al. DeepSeMS: a large language model reveals hidden biosynthetic potential of the global ocean microbiome.
# ----------------------------------------------------------------
import numpy as np
import pandas
import argparse
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import KFold 
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def scaffold_aligned_enumeration(smiles, smiles_scaffold):   
    scaffold=Chem.MolFromSmiles(smiles_scaffold)
    if scaffold is None:
        return(smiles)
    mol=Chem.MolFromSmiles(smiles)
    mol_atom_list=list(range(mol.GetNumAtoms()))    
    matches=mol.GetSubstructMatches(scaffold)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/data_set.csv')
    parser.add_argument('--output', type=str, default='./data/')
    parser.add_argument('--type', type=int, choices=[0,1], default=0, help='0: structural features-aligned SMILES enumeration, 1: randomized SMILES enumeration')
    parser.add_argument('--enum_factor', type=int, default=100)
    parser.add_argument('--max_tries', type=int, default=500)
    args = parser.parse_args()
    enum_factor = args.enum_factor
    max_tries = args.max_tries
    output_path = args.output
    table_columns = ["BGC_features", "SMILES"]
    data_amplified = []
    df = pd.read_csv(args.input, sep="\t")
    bgc_features, smiles = df['BGC_features'].values, df['SMILES'].values
    for (bgc_feature, smi) in tqdm(zip(bgc_features, smiles)):
        tries = []
        m = Chem.MolFromSmiles(smi)
        if args.type == 0:
            m_scaffold = MurckoScaffold.GetScaffoldForMol(m)
            smi_scaffold = Chem.MolToSmiles(m_scaffold, isomericSmiles=False, kekuleSmiles=True)
            if len(smi_scaffold) > 0:
                for try_idx in range(max_tries):
                    this_try = scaffold_aligned_enumeration(smi, smi_scaffold)
                    tries.append(this_try)
                    tries = [rnd for rnd in np.unique(tries)]
                    if len(tries) > enum_factor:
                        tries = tries[:enum_factor]
                        break
        else:
            for try_idx in range(max_tries):
                for try_idx in range(max_tries):
                    this_try = Chem.MolToSmiles(m, doRandom=True, canonical=False, isomericSmiles=False, kekuleSmiles=True)
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
            data_amplified.append(datapair)

    res_df = pandas.DataFrame(data_amplified)
    res_df = res_df[table_columns]
    res_df = res_df.sample(frac=1).reset_index(drop=True)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)  

    tran_data_files = [output_path + f"tran_{i}.csv" for i in range(10)] 
    val_data_files = [output_path + f"val_{i}.csv" for i in range(10)]  

    for fold, (train_idx, val_idx) in enumerate(kf.split(res_df)):  
        train_df = res_df.iloc[train_idx]  
        val_df = res_df.iloc[val_idx]
        train_df = train_df[table_columns]
        val_df = val_df[table_columns]
        train_df.to_csv(tran_data_files[fold], sep="\t", header=1, index=False, columns=table_columns)  
        val_df.to_csv(val_data_files[fold], sep="\t", header=1, index=False)  