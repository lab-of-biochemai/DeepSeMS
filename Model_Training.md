### DeepSeMS Model Training
Model training is optional and is only required if users wish to reproduce the published model or retrain DeepSeMS on new datasets or with customized data preprocessing and augmentation.

### Table of Contents
- [Prepare Training Data](#prepare-training-data)
  - [Option 1: Processing from raw data](#option-1-processing-from-raw-data)
    - [Install additional dependency](#install-additional-dependency)
    - [Prepare raw data](#prepare-raw-data)
    - [Data processing and augmentation](#data-processing-and-augmentation)
  - [Option 2: Using the processed training dataset](#option-2-using-the-processed-training-dataset)
- [Model Training](#model-training)
- [Customized Training](#customized-training)

### Prepare Training Data
#### Option 1: Processing from raw data
Intended for advanced users who wish to retrain DeepSeMS on new datasets or with customized data preprocessing and augmentation.
##### Install additional dependency
```Bash
pip install scikit-learn==1.7.2
```
##### Prepare raw data
Prepare raw data with BGC-SMILES pairs as the same format as the example file provided in the repository: `./data/data_set.csv`.  
- Each row corresponds to one BGC and its associated molecular structure, represented by:
	-	`BGC_features`: An ordered Python list of Pfam identifiers.
	-	`SMILES`: The corresponding molecular structure encoded as a SMILES string.
##### Data processing and augmentation
Run `data_processing.py` to process the data set for training. It will perform data augmentation, SMILES canonicalization, and data partitioning.
```Bash
python data_processing.py
```
- Arguments:
  - `--input`: Path to the data set file. (default: ./data/data_set.csv)
  - `--output`: Directory to save the output file. (default: ./data/)
  - `--type`: Data augmentation type. Options: 0 (structural features-aligned SMILES enumeration) or 1 (randomized SMILES enumeration). (default: 0)
  - `--enum_factor`: Data amplification factor. (default: 100)
  - `--max_tries`: Maximum trying number for SMILES enumeration. (default: 500)

⚠️ Important: The data augmentation may introduce randomness and lead to different performance between the retrained model and the published DeepSeMS model.
#### Option 2: Using the processed training dataset
Uses the curated, fully processed training dataset released by the authors and reproduces the model reported in the manuscript and used by the web server. 
You can download the processed training data (https://figshare.com/ndownloader/files/60134648) for reproducing the published DeepSeMS model.   
Unzip and place the training data files (e.g., `tran_*.csv, val_*.csv`) into the `./data` directory.

### Model Training
To retrain the model using the default hyperparameters (10-fold Cross-Validation), simply run the `train.py` script without arguments.   
⚠️ It will take 5-6 days on one NVIDIA-GeForce-RTX-4090 GPU:
```Bash
python train.py
```
This will train 10 separate models and save checkpoints (`checkpoint0.ckpt` ... `checkpoint9.ckpt`) to the `./checkpoints/` folder.

### Customized Training
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
