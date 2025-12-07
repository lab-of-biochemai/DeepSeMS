# DeepSeMS

**DeepSeMS** is a large language model (LLM) based on Transformer architecture designed to reveal the hidden biosynthetic potential of the global ocean microbiome. It characterizes chemical structures of natural molecules produced by microbes directly from biosynthetic gene clusters (BGCs).

## Table of Contents
- [Publication](#publication)
- [Web Server](#web-server)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
  - [Option 2: Local Environment](#option-2-local-environment)
- [Data Preparation](#data-preparation)
- [Usage (Prediction)](#usage-prediction)
- [Model Training](#model-training)
- [Requirements](#requirements)

## Publication
*In submission.*

## Web Server
For a quick test without installation, visit our web server:
[https://biochemai.cstspace.cn/deepsems/](https://biochemai.cstspace.cn/deepsems/)

## Project Structure
Ensure your local directory is organized as follows before running the model:
```text
DeepSeMS/
├── checkpoints/       # Place model weights here (.ckpt)
├── data/
│   ├── pfam/          # Place Pfam files here
│   ├── tran_0.csv ... tran_9.csv (for 10-fold CV)
│   └── val_0.csv  ... val_9.csv
├── vocabs/            # Vocabulary files
├── test/              # Input files for prediction
│   ├── outputs/       # Annotation files
├── tokenizer/
│   ├── tokenizer.py   # Tokenizer 
├── models/            # Model architecture code
├── predict.py         # Prediction script
├── train.py           # Training script
└── README.md
```
## Installation & Setup
### Option 1: Docker (Recommended)
We provide a pre-configured Docker image with all dependencies installed.
#### Step 1: Pull the DeepSeMS docker image
```bash
docker pull tingjunxu2022/deepsems:v1
```
#### Step 2: Download required data
Before running the container, you must download the necessary model weights and database files to your local machine (see [Data Preparation](#data-preparation) section).
#### Step 3: Run the container
Mount your current directory (containing code and data) to `/deepsems` inside the container.
```Bash
# Assuming the DeepSeMS project directory is /home/user/deepsems.
docker run -it -v /home/user/deepsems:/deepsems tingjunxu2022/deepsems:v1 /bin/bash
# Inside the container:
cd /deepsems
```
Or run the container with GPUs. ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
```Bash
# Assuming the DeepSeMS project directory is /home/user/deepsems.
docker run --gpus all -it -v /home/user/deepsems:/deepsems tingjunxu2022/deepsems:v1 /bin/bash
# Inside the container:
cd /deepsems
```
### Option 2: Local Environment
If you prefer to run it locally without Docker, we recommend using Conda.
#### Step 1: Create a Conda environment
```Bash
conda create -n deepsems python=3.10
conda activate deepsems
```
#### Step 2: Install HMMER
```Bash
conda install -c bioconda hmmer=3.3.2
```
#### Step 3: Install Python dependencies
```Bash
pip install torch==2.1.0 torchtext==0.16.0
pip install biopython==1.79 pandas==2.0.3 rdkit==2023.03.1
```
### Data Preparation
You must download the support files before running predictions or training.
#### Model Checkpoints:
Download from: http://doi.org/10.6084/m9.figshare.29680658.  
Place `.ckpt` files into the `./checkpoints/` directory.
#### Pfam Database:
Download `Pfam.zip` from the link above.  
Place the Pfam files into the `./data/pfam/` directory.  
Note: You may need to run `hmmpress ./data/pfam/Pfam-A.hmm` if index files are missing.
#### Training Data (Only for retraining):
Download `training_data.zip` from the link above.  
Place CSV files (e.g., `tran_*.csv, val_*.csv`) into the `./data` directory.  
## Usage (Prediction)
Use `predict.py` to generate SMILES strings from gene clusters.
### 1. Predict from antiSMASH results (GenBank)
This is the default mode. It extracts biosynthetic genes from a `.gbk` file.
```Bash
python predict.py --input ./test/deepsems_sample.gbk --type antismash
```
```Bash
Arguments:
--input: Path to the input file.
--type: Input format. Options: antismash (default) or deepbgc.
--output: Directory to save annotation files (default: ./test/outputs/).
```
### 2. Predict from DeepBGC results (FASTA)
If you have a `FASTA` file containing protein sequences (e.g., from DeepBGC).
```Bash
python predict.py --input ./test/DeepBGC_sample.fa --type deepbgc
```
```Bash
Arguments:
--input: Path to the input file.
--type: Input format. Options: antismash (default) or deepbgc.
--output: Directory to save annotation files (default: ./test/outputs/).
```
#### Sample result:
Results are ranked by consensus across the top-10 models and predicted scores, with the top-ranked structure being the one most consistently predicted.
```Bash
Top Predictions: 
------------------------------
Rank: 1 
Predicted SMILES: CC(C)C1C=CC(=O)NCCC=CC(NC(=O)C(NC(=O)O)C(C)C)C(=O)N1
Predicted score: 87.86
Consensus count: 5
------------------------------
Rank: 2
Predicted SMILES: CCCCCCC=CC=CC(=O)NC(C(=O)NC1CCCCNC(=O)C=CC(CC)NC1=O)C(C)O
Predicted score: 85.68
Consensus count: 3
------------------------------
Rank: 3
Predicted SMILES: CCCCCCCCC=CCC(=O)NC(C(=O)NC1CC(=O)C=CC(C(C)C)NC1=O)C(C)C
Predicted score: 83.41
Consensus count: 2
...
```
## Model Training
### 1. Retraining (Default)
To retrain the model using the default hyperparameters (10-fold Cross-Validation), simply run the script without arguments.  
Ensure training data (`tran_0.csv` to `tran_9.csv` and `val_0.csv` to `val_9.csv`) is in `./data/`.  
Run the training script:
```Bash
python train.py
```
This will train 10 separate models and save checkpoints (`checkpoint0.ckpt` ... `checkpoint9.ckpt`) to the `./checkpoints/` folder.
### 2. Customized Training
You can customize the model architecture and training process.  
Below is the full list of arguments available you can pass to the `train.py` script for customization:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Training Setup** | | | |
| `--batch_size` | `int` | `64` | Number of samples per batch. |
| `--epochs` | `int` | `500` | Total number of training epochs per fold. |
| `--lr` | `float` | `0.0001` | Learning rate for the AdamW optimizer. |
| `--patience` | `int` | `10` | Early stopping patience (epochs without improvement). |
| **Model Architecture** | | | |
| `--d_model` | `int` | `512` | Dimension of the embeddings and hidden layers. |
| `--n_heads` | `int` | `8` | Number of attention heads. |
| `--n_enc` | `int` | `6` | Number of encoder layers. |
| `--n_dec` | `int` | `6` | Number of decoder layers. |
| `--dropout` | `float` | `0.1` | Dropout probability. |
| **Misc** | | | |
| `--model_prefix` | `str` | `checkpoint` | Prefix for saved model files (e.g., `checkpoint0.ckpt`). |

## Requirements
Annotated versions are tested, later versions should generally work.
- Language: `Python 3.10`
- Deep Learning: `PyTorch 2.1.0, TorchText 0.16.0`
- Bioinformatics: `HMMER3 (v3.3.2), Biopython (v1.79), Pfam Database (v36.0)`
- Chemistry: `RDKit (v2023.03.1)`
- Data Handling: `Pandas (v2.0.3)`
## Preferred Hardware
- `CUDA 12.0` (tested)
- `GPU VRAM: 24 GB` (NVIDIA GeForce RTX 4090 tested)

