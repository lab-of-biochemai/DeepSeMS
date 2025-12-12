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

import torch
from Bio import SearchIO
from Bio import SeqIO
import os
from torchtext.vocab import build_vocab_from_iterator
import argparse
import pandas as pd
from rdkit import Chem, RDLogger
import numpy as np
import subprocess
from models.Transformer import Transformer
from collections import Counter

RDLogger.DisableLog("rdApp.*")
os.chdir(os.path.dirname(os.path.realpath(__file__)))
tokenize = lambda x : list(x)


def get_vocab_bgc(train_datapipe):
    vocab = build_vocab_from_iterator(yield_tokens_bgc(train_datapipe),
                                     specials=['<UNK>', '<PAD>', '<SOS>', '<EOS>'],
                                     max_tokens=20000)
    vocab.set_default_index(vocab['<UNK>'])
    return vocab


def yield_tokens_bgc(data_iter):
    for text in data_iter:
        yield [text]


def tokenizer_bgc_features(bgc_features):
    tokens = eval(bgc_features)
    tokens = ['<SOS>'] + tokens + ['<EOS>']
    return tokens


def predict(model, input_sequence, max_length=250, PAD_token=1, SOS_token=2, EOS_token=3, device='cpu'):
    model.eval()
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
    length_penalty = 0.6
    sum_logprobs = 0
    for _ in range(max_length):
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        pred = model(input_sequence, y_input, tgt_mask)
        # num with highest probability
        next_item = pred.topk(1)[1].view(-1)[-1].item() 
        next_item = torch.tensor([[next_item]], device=device)
        probs = pred.topk(1)[0].view(-1)[-1].item()
        sum_logprobs = sum_logprobs + probs
        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)
        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token or next_item.view(-1).item() == PAD_token:
            break
    score = sum_logprobs / len(y_input.view(-1).tolist()) ** length_penalty
    return y_input.view(-1).tolist(), score


def bgc_features_to_numbers(bgc_features, bgc_features_voc):
    return [bgc_features_voc[token] for token in tokenizer_bgc_features(bgc_features)]


def smiles_to_string(smiles, smiles_vocab):
    return ''.join([smiles_vocab.get_itos()[word] for word in smiles])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict SMILES from BGC data (antiSMASH gbk or DeepBGC fasta).")
    parser.add_argument('--input', type=str, required=True, help='Path to input file (.gbk for antismash, .fa for deepbgc)')
    parser.add_argument('--type', type=str, default='antismash', choices=['antismash', 'deepbgc'], 
                        help='Input file type: "antismash" (default) or "deepbgc"')
    parser.add_argument('--output', type=str, default='./test/outputs/', help='Output directory')
    parser.add_argument('--pfam', type=str, default='./data/pfam/', help='Pfam database files directory')
    parser.add_argument('--max', type=int, default=250, help='Max sequence length')
    parser.add_argument('--pad', type=int, default=1, help='PAD token ID')
    parser.add_argument('--sos', type=int, default=2, help='SOS token ID')
    parser.add_argument('--eos', type=int, default=3, help='EOS token ID')
    
    args = parser.parse_args()
    
    # ---------------- Setup Paths and Device ----------------
    ckpt_path = "./checkpoints/"
    root = os.path.dirname(__file__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---------------- Load Vocabularies ----------------
    print("Loading vocabularies...")
    bgc_features = pd.read_csv('./vocabs/bgc_features_vacab.csv')['bgc_features'].tolist()
    bgc_features_voc = get_vocab_bgc(bgc_features)
    smiles_vocab = torch.load('./vocabs/smiles-vocab.pt')

    # ---------------- Output Directory Setup ----------------
    jobname = os.path.basename(args.input)
    if args.type == 'antismash':
        jobname = jobname.replace(".gbk", "")
    else:
        jobname = jobname.replace(".fa", "")
        
    output_path = os.path.join(args.output, jobname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    annotation_path = os.path.join(output_path, "annotation")
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)

    # ---------------- Preprocessing (Extract Protein Sequence) ----------------
    protein_fasta_path = ""
    
    if args.type == 'antismash':
        print(f"Processing antiSMASH input: {args.input}")
        input_handle = open(args.input, "r", encoding='UTF-8')
        protein_fasta_path = os.path.join(annotation_path, jobname + ".fa")
        output_handle = open(protein_fasta_path, "w")

        try:
            records = list(SeqIO.parse(input_handle, "genbank"))
            if not records:
                print("No GenBank records found.")
                exit()
            record = records[0]
            
            i = 0
            for fe in record.features:
                gene = None
                gene_kind = None
                gene_functions = None
                product = None
                protein_id = None
                seqs = None
                locus_tag = None
                
                if fe.type == "CDS" and "gene_kind" in fe.qualifiers.keys():
                    gene_kind = fe.qualifiers["gene_kind"][0]
                    if gene_kind == "biosynthetic" or gene_kind == "biosynthetic-additional":
                        i = i + 1
                        if "gene" in fe.qualifiers.keys():
                            gene = fe.qualifiers["gene"][0]
                        if "protein_id" in fe.qualifiers.keys():
                            protein_id = fe.qualifiers["protein_id"][0]
                        if "locus_tag" in fe.qualifiers.keys():
                            locus_tag = fe.qualifiers["locus_tag"][0]
                        
                        if "translation" in fe.qualifiers:
                            seqs = fe.qualifiers["translation"][0]
                            if protein_id is not None:
                                output_handle.write(">%s\n%s\n" % (protein_id, seqs))
                            elif locus_tag is not None:
                                output_handle.write(">%s\n%s\n" % (locus_tag, seqs))
                            elif gene is not None:
                                output_handle.write(">%s\n%s\n" % (gene, seqs))
                            else:
                                output_handle.write(">%s\n%s\n" % ("biosyn-"+ str(i), seqs))
            print(f"Annotated biosynthetic enzymes: {i}")
        except Exception as e:
            print(f"Error parsing GenBank file: {e}")
            exit()
        finally:
            output_handle.close()
            input_handle.close()

    elif args.type == 'deepbgc':
        print(f"Processing DeepBGC input: {args.input}")
        protein_fasta_path = args.input

    # ---------------- HMMScan ----------------
    print("Running HMMScan...")
    hmmdb = args.pfam + "Pfam-A.hmm"
    hmmoutpath = os.path.join(annotation_path, jobname + ".domtblout.txt")
    
    p = subprocess.Popen(
        ['hmmscan', '--nobias', '--domtblout', hmmoutpath, hmmdb, protein_fasta_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    out, err = p.communicate()

    if p.returncode or not os.path.exists(hmmoutpath):
        print("Error in HMMScan:")
        print(err.strip())
        
    # ---------------- Parse HMM Results ----------------
    print("Parsing HMM results...")
    try:
        queries = SearchIO.parse(hmmoutpath, 'hmmscan3-domtab')
    except Exception as e:
        print(f"Error parsing HMM output: {e}")
        queries = []

    pfam_ids_bgc = []
    for query in queries:
        for hit in query.hits:
            best_index = np.argmin([hsp.evalue for hsp in hit.hsps])
            best_hsp = hit.hsps[best_index]
            pfam_id = hit.accession
            evalue = float(best_hsp.evalue)
            
            if evalue > 0.01:
                continue
                
            pfam_ids_bgc.append(pfam_id)

    print(f"Annotated BGC features count: {len(pfam_ids_bgc)}")
    # print(pfam_ids_bgc)
    bgc_feature_str = str(pfam_ids_bgc)

    # ---------------- Model Prediction ----------------
    print("Loading Model...")
    model = Transformer(
        src_tokens=1020,
        trg_tokens=35, 
        dim_model=512, 
        num_heads=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dropout_p=0.1
    ).to(device)

    print("Predicting...")
    results = []
    for i in range(10):
        result = {}
        ckpt_name = 'checkpoint' + str(i)
        ckpt = os.path.join(ckpt_path, ckpt_name + ".ckpt")
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found at {ckpt}")
            exit()

        model_dict = torch.load(ckpt, map_location=torch.device(device))
        model.load_state_dict(model_dict["model"])
        model.eval()

        input_tensor = torch.tensor([bgc_features_to_numbers(bgc_feature_str, bgc_features_voc)], dtype=torch.long, device=device)
        
        output, score = predict(model, input_tensor, args.max, args.pad, args.sos, args.eos, device=device)
        result_smiles = smiles_to_string(output[1:-1], smiles_vocab)
        try:
            m = Chem.MolFromSmiles(result_smiles)
        except:
            continue
        if m != None:
            result['smi'] = Chem.MolToSmiles(m, canonical=True)
            result['score']= round(score,2)
            results.append(result)
    if len(results) == 0:
        print("No valid SMILES predicted.")
        exit()
    else:
        count_dict = Counter(item['smi'] for item in results)
        smis = []
        results_con = []
        output_results = []
        output_table = os.path.join(output_path, jobname + "_result.csv")
        table_columns = ["Rank", "Predicted SMILES", "Predicted score", "Consensus count"]
        for result in results:
            smi = Chem.CanonSmiles(result['smi'], useChiral=0)
            result['consensus'] = count_dict[result['smi']]
            if not smi in smis:
                smis.append(smi)
                results_con.append(result)

        results_con = sorted(results_con, key=lambda x: (-x['consensus'], -x['score']))
        print("\nTop Predictions:")
        for idx, res in enumerate(results_con):
            print("-" * 30)
            print(f"Rank: {idx + 1}")
            print(f"Predicted SMILES: {res['smi']}")
            print(f"Predicted score: {res['score']:.2f}")
            print(f"Consensus count: {res['consensus']}")
            output_result = {}
            output_result["Rank"] = idx + 1
            output_result["Predicted SMILES"] = res['smi']
            output_result["Predicted score"] = f"{res['score']:.2f}"
            output_result["Consensus count"] = res['consensus']
            output_results.append(output_result)
        # ---------------- Save Results ----------------
        res_df = pd.DataFrame(output_results)
        res_df = res_df[table_columns]
        res_df.to_csv(output_table, sep="\t", header=True, index=False, mode='w')