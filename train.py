# ----------------------------------------------------------------
# Large language model revealing hidden biosynthetic potential of the microbiomes
# ----------------------------------------------------------------
# References: 
# https://github.com/huggingface/transformers; 
# https://github.com/DSPsleeporg/smiles-transformer; 
# https://github.com/yydiao1025/Macformer
# https://github.com/Merck/deepbgc
# https://github.com/pschwllr/MolecularTransformer
# https://github.com/gmattedi/SmilesTransformer
import torch
import numpy as np
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from tokenizer import RegexTokenizer
from tqdm import tqdm
import random
from rdkit.Chem import MACCSkeys
from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")  # Disable RDKit warnings

# Change working directory to the directory of the script
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# Load the vocabulary for SMILES
smiles_vocab = torch.load('./vocabs/smiles-vocab.pt')
# Define a simple tokenizer function
tokenize = lambda x : x.split()
# Initialize the RegexTokenizer
tokenizer = RegexTokenizer()

# Function to build the vocabulary for BGC features
def get_vocab_bgc(train_datapipe):
    vocab = build_vocab_from_iterator(yield_tokens_bgc(train_datapipe),
                                     specials=['<UNK>', '<PAD>', '<SOS>', '<EOS>'],
                                     max_tokens=20000)
    vocab.set_default_index(vocab['<UNK>'])
    return vocab

# Function to yield tokens for BGC features
def yield_tokens_bgc(data_iter):
    for text in data_iter:
        yield tokenize(text)

# Function to tokenize BGC features
def tokenizer_bgc_features(bgc_features):
    tokens = eval(bgc_features)
    return tokens

# Function to process the data
def data_process(data_path):
    df = pd.read_csv(data_path, sep="\t")
    src_bgc_features, tgt_smiles = df['BGC_features'].values, df['SMILES'].values
    data = []
    for (bgc_feature, smile) in zip(src_bgc_features, tgt_smiles):
        smile_tensor_ = np.array([smiles_vocab[token] for token in tokenizer.tokenize(smile)]).astype(np.int64)
        bgc_feature_tensor_ = np.array([bgc_features_voc[token] for token in tokenizer_bgc_features(bgc_feature)]).astype(np.int64)
        data.append([bgc_feature_tensor_, smile_tensor_])
    random.shuffle(data)
    return data

# Function to generate batches
def generate_batch(data, batch_size, padding_token):
    batches = []
    d = data
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size-1 < len(data):
            for i in range(2):
                max_batch_length = 0

                # Get longest sentence in batch
                for seq_pack in data[idx : idx + batch_size]:
                    if len(seq_pack[i]) > max_batch_length:
                        max_batch_length = len(seq_pack[i])

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx][i])
                    padding = [padding_token] * remaining_length
                    data[idx + seq_idx][i] = np.concatenate((SOS_token, data[idx + seq_idx][i], padding, EOS_token))
            # print(d[idx : idx + batch_size])
            batches.append(np.array(d[idx : idx + batch_size]))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches

# Class for Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

# Class for Transformer model
class Transformer(nn.Module):

    # Constructor
    def __init__(
        self,
        src_tokens,
        trg_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.input_embedding = nn.Embedding(src_tokens, dim_model)
        self.output_embedding = nn.Embedding(trg_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, trg_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):

        src_emb = self.input_embedding(src) * math.sqrt(self.dim_model)
        tgt_emb = self.output_embedding(tgt) * math.sqrt(self.dim_model)
        src_emb = self.positional_encoder(src_emb)
        tgt_emb = self.positional_encoder(tgt_emb)

        # Transformer blocks - Out size = (sequence length, batch_size, dim_model)
        transformer_out = self.transformer(src_emb.transpose(0, 1), tgt_emb.transpose(0, 1), tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        transformer_out = transformer_out.transpose(0, 1)


        # Output layer - Out size = (batch_size, sequence length, output_vocab_size)
        out = self.out(transformer_out)
        out = out.permute(1,0,2)
        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

# Function for training loop
def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0

    ####
    total_correct = 0
    total_samples = 0
    ####

    for batch in tqdm(dataloader):
        X, y = batch[:, 0], batch[:, 1]
        X, y = np.vstack(X).astype(np.int64), np.vstack(y).astype(np.int64)
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
    
    #######
        # Calculate accuracy
        _, predicted = torch.max(pred.data, 1)
        total_samples += y_expected.size(0) * y_expected.size(1)
        total_correct += (predicted == y_expected).sum().item()

    accuracy = total_correct / total_samples

    return total_loss / len(dataloader), accuracy
    #####

    # return total_loss / len(dataloader)

# Function for validation loop
def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    ####
    total_correct = 0
    total_samples = 0
    ####
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = np.vstack(X).astype(np.int64), np.vstack(y).astype(np.int64)
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)

            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
    #######
        # Calculate accuracy
        _, predicted = torch.max(pred.data, 1)
        total_samples += y_expected.size(0) * y_expected.size(1)
        total_correct += (predicted == y_expected).sum().item()

    accuracy = total_correct / total_samples

    return total_loss / len(dataloader), accuracy
    #####
    # return total_loss / len(dataloader)


# Function to fit the model
def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, patience, model_name):
    train_loss_list, validation_loss_list, train_acc_list, validation_acc_list = [], [], [], []

    df = pd.read_csv('./data/validation_data.csv', sep='\t')
    src_bgc_features, tgt_smiles = df['BGC_features'].values, df['SMILES'].values

    test_smi_list = []
    counter = 0
    fpgen = AllChem.GetRDKitFPGenerator()
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)

        train_loss, train_acc = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        train_acc_list += [train_acc]

        validation_loss, validation_acc = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        validation_acc_list += [validation_acc]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {validation_acc:.4f}")

        model_state_dict = model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "epoch": epoch}
        model_path = ("./checkpoints/" + model_name + ".ckpt")

        if validation_loss <= min(validation_loss_list):
            torch.save(checkpoint, model_path)
            print("Checkpoint file updated")
            counter = 0
            last_epoch = epoch + 1
        else:
            counter += 1
            print(f"Validation loss not improved, patience: {counter}")
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best epoch {last_epoch}")
                break           
        print()

    return train_loss_list, validation_loss_list, train_acc_list, validation_acc_list

# Load BGC features
bgc_features = pd.read_csv(r'./vacabs/bgc_features_vacab.csv')['bgc_features'].tolist()
# Build vocabulary for BGC features
bgc_features_voc = get_vocab_bgc(bgc_features)

# Function to convert SMILES to string
def smiles_to_string(smiles, smiles_vocab):
    return ''.join([smiles_vocab.get_itos()[word] for word in smiles])

# Function to convert BGC features to numbers
def bgc_features_to_numbers(bgc_features):
    return [bgc_features_voc[token] for token in tokenizer_bgc_features_val(bgc_features)]

# Function to tokenize BGC features for validation
def tokenizer_bgc_features_val(bgc_features):
    tokens = eval(bgc_features)
    tokens = ['<SOS>'] + tokens + ['<EOS>']
    return tokens

# Function to predict
def predict(model, input_sequence, max_length=250, PAD_token=1, SOS_token=2, EOS_token=3):
    model.eval()
    
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token or next_item.view(-1).item() == PAD_token:
            break

    return y_input.view(-1).tolist()

# Main function
if __name__ == '__main__':
    model_name = 'checkpoint'
    data_path = './data/training_data.csv'
    val_path = './data/validation_data.csv'
    data = data_process(data_path)
    val_data = data_process(val_path)

    SOS_token = np.array([2])
    EOS_token = np.array([3])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = Transformer(
        src_tokens=1020, trg_tokens=35, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1
    ).to(device)

    train_batch = generate_batch(data, 64, 1)
    val_batch = generate_batch(val_data, 303, 1)

    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_list, validation_loss_list, train_acc_list, validation_acc_list = fit(model, opt, loss_fn, train_batch, val_batch, 500, 10, model_name)
    plt.plot(train_loss_list, label='Train loss')
    plt.plot(validation_loss_list, label='Validation loss')
    plt.show()