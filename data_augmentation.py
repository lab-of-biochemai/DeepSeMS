# ----------------------------------------------------------------
# TJ Xu et al. DeepSeMS: a large language model reveals hidden biosynthetic potential of the global ocean microbiome.
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
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import random


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
    parser.add_argument('--output', type=str, default='./data/data_set_amplified.csv')
    parser.add_argument('--type', type=int, choices=[0,1], default=0, help='0: randomized SMILES enumeration, 1: structural features-aligned SMILES enumeration')
    parser.add_argument('--enum_factor', type=int, default=100)
    parser.add_argument('--max_tries', type=int, default=500)
    args = parser.parse_args()
    enum_factor = args.enum_factor
    max_tries = args.max_tries
    datatable = args.output
    output_path = Path(datatable)
    table_columns = ["BGC_features", "SMILES"]
    output_path.write_text("\t".join(table_columns) + "\n")
    trainingdata_amplified = []
    df = pd.read_csv(args.input, sep="\t")
    bgc_features, smiles = df['BGC_features'].values, df['SMILES'].values
    for (bgc_feature, smi) in zip(bgc_features, smiles):
        tries = []
        m = Chem.MolFromSmiles(smi)
        if args.type == 0:
            for try_idx in range(max_tries):
                for try_idx in range(max_tries):
                    this_try = Chem.MolToSmiles(m, doRandom=True, canonical=False, isomericSmiles=False, kekuleSmiles=True)
                    tries.append(this_try)
                    tries = [rnd for rnd in np.unique(tries)]
                    if len(tries) > enum_factor:
                        tries = tries[:enum_factor]
                        break
        else:
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
    if len(tries) == 0:
        tries.append(smi)
    for sms in tries:
        datapair = {}
        datapair["BGC_features"] = bgc_feature
        datapair["SMILES"] = sms
        trainingdata_amplified.append(datapair)

    res_df = pandas.DataFrame(trainingdata_amplified)
    res_df = res_df[table_columns]
    train_df = res_df.sample(frac=1)
    res_df.to_csv(output_path, sep="\t", header=False, index=False, mode='a')