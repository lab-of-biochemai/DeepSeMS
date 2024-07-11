# ----------------------------------------------------------------
# Large language model revealing hidden biosynthetic potential of the deep ocean microbiome
# ----------------------------------------------------------------
# References: 
# https://github.com/huggingface/transformers; 
# https://github.com/DSPsleeporg/smiles-transformer; 
# https://github.com/yydiao1025/Macformer
# https://github.com/Merck/deepbgc
# https://github.com/pschwllr/MolecularTransformer
# https://github.com/gmattedi/SmilesTransformer
import torch
import time
from Bio import SearchIO
from Bio import SeqIO
import os
from torchtext.vocab import build_vocab_from_iterator
import argparse
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import numpy as np
import subprocess
from models.Transformer import Transformer


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


def predict(model, input_sequence, max_length=250, PAD_token=1, SOS_token=2, EOS_token=3):
    model.eval()
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
    length_penalty = 0.6
    num_tokens = len(input_sequence[0])
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./test/testbgc.gbk')
    parser.add_argument('--output', type=str, default='./test/outputs/')
    parser.add_argument('--max', type=int, default=250)
    parser.add_argument('--pad', type=int, default=1)
    parser.add_argument('--sos', type=int, default=2)
    parser.add_argument('--eos', type=int, default=3)
    args = parser.parse_args()
    start = time.time()
    ckpt_name = 'checkpoint'
    ckpt_path = "./checkpoints/"

    if args.input == 'none':
        print("Please provide an --input argument.")
        exit()
    jobname = os.path.basename(args.input)
    jobname = jobname.replace(".gbk","")
    output_path = args.output + "/" + jobname
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    annotation_path = output_path + "/annotation"
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)

    input_handle = open(args.input, "r", encoding='UTF-8')
    output_handle = open(annotation_path + "/" + jobname + ".fa", "w")
    record = list(SeqIO.parse(input_handle, "genbank"))
    record = record[0]
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
                seqs = fe.qualifiers["translation"][0]
                if protein_id is not None:
                    output_handle.write(">%s\n%s\n" % (protein_id, seqs))
                elif locus_tag is not None:
                    output_handle.write(">%s\n%s\n" % (locus_tag, seqs))
                elif gene is not None:
                    output_handle.write(">%s\n%s\n" % (gene, seqs))
                else:
                    output_handle.write(">%s\n%s\n" % ("biosyn-"+ str(i), seqs))
    output_handle.close()
    input_handle.close()
    print(f"Annotated biosynthetic enzymes: {i}")

    hmmdb = "./data/pfam/Pfam-A.hmm"
    proteinpath = annotation_path + "/" + jobname + ".fa"
    hmmoutpath = os.path.join(annotation_path, jobname + ".domtblout.txt")
    p = subprocess.Popen(
        ['hmmscan', '--nobias', '--domtblout', hmmoutpath, hmmdb, proteinpath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    out, err = p.communicate()
    if p.returncode or not os.path.exists(hmmoutpath):
        print(err.strip())
    queries = SearchIO.parse(hmmoutpath, 'hmmscan3-domtab')
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
    print(f"Annotated BGC features: {len(pfam_ids_bgc)}")

    bgc_feature = str(pfam_ids_bgc)
    root = os.path.dirname(__file__)
    bgc_features = pd.read_csv('./vocabs/bgc_features_vacab.csv')['bgc_features'].tolist()
    bgc_features_voc = get_vocab_bgc(bgc_features)
    smiles_vocab = torch.load('./vocabs/smiles-vocab_c.pt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = Transformer(
        src_tokens=1020,
        trg_tokens=35, 
        dim_model=512, 
        num_heads=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dropout_p=0.1
    ).to(device)
    ckpt = ckpt_path + ckpt_name + ".ckpt"
    model_dict = torch.load(ckpt, map_location=torch.device(device))
    model.load_state_dict(model_dict["model"])
    model.eval()

    input = torch.tensor([bgc_features_to_numbers(bgc_feature, bgc_features_voc)], dtype=torch.long, device=device)
    result, score = predict(model, input, args.max, args.pad, args.sos, args.eos)
    result = smiles_to_string(result[1:-1], smiles_vocab)
    print(f"Predicted SMILES: {result}")
    print(f"Predicted score: {score:.2f}")
    try:
        m = Chem.MolFromSmiles(result)
    except:
        m = None
    if m != None:
        print(f"Valid: True.")
    else:
        print("Valid: False.")
    i = 1
    end = time.time()
    print(f"Running time: {end-start:.2f} Seconds")
        

