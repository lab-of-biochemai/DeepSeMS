# References: 
# https://github.com/pschwllr/MolecularTransformer
# https://github.com/gmattedi/SmilesTransformer
import re
from typing import Sequence, Dict, List, Tuple
import numpy as np

"""
Define some constants
"""
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

# TODO cite
SMI_REGEX_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN) -> None:
        """Constructs a RegexTokenizer.
        Args:
            regex_pattern: regex pattern used for tokenization.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str, bos_token=BOS_WORD, eos_token=EOS_WORD) -> List[str]:
        """
        Tokenize input text

        Args:
            text (str): Text
            bos_token (str): Beginning of sentence token
            eos_token (str): End of sentence token

        Returns:
            tokens (List[str]): Tokenized input

        """
        tokens = [token for token in self.regex.findall(text)]
        # tokens = ['<SOS>'] + tokens + ['<EOS>']
        return tokens


def build_vocabulary(smiles_lst: Sequence[str], min_occurrences: int = 5) -> Dict[str, int]:
    """
    Build a token -> index mapping given a list of smiles string.
    Uses SmilesTransformer.tokenizer.RegexTokenizer

    Args:
        smiles_lst (Sequence[str])
        min_occurrences (int): Minimum number of occurrences for the token
            to be added to the vocabulary

    Returns:
        token2idx (Dict[str, int]): Token -> index mapping

    """
    tokenizer = RegexTokenizer()

    tokens = [tokenizer.tokenize(smi) for smi in smiles_lst]
    unique_tokens, unique_counts = np.unique(np.concatenate(tokens), return_counts=True)

    token2idx = {}

    i = 0
    for t, c in zip(unique_tokens, unique_counts):
        if c <= min_occurrences:
            continue
        else:
            token2idx[t] = i
            i += 1

    special_tokens = [BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD]
    for t in special_tokens:
        if t not in token2idx:
            token2idx[t] = token2idx.get(t, i)
            i += 1

    return token2idx


def load_mapping(alphabet_path: str, sep: str = "\n") -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Return Token -> index and index -> token mapping from a file containing the alphabet

    Args:
        alphabet_path (str): `sep`-separated file containing the list of token
        sep (str): Token separator in the file

    Returns:
        token2idx (Dict[str, int]): Token -> index mapping
        idx2token (Dict[int, str]): Index -> token mapping

    """
    with open(alphabet_path) as handle:
        alphabet = handle.read().split(sep)

    token2idx = {token: i for i, token in enumerate(alphabet)}
    idx2token = {i: token for token, i in token2idx.items()}

    return token2idx, idx2token


def dense_onehot(tokens: Sequence[str], token2idx: Dict[str, int]) -> List[int]:
    """
    Map tokens to idx sequence.
    If a token is not found in `token2idx`, then the UNK_WORK index is used
    Args:
        tokens (Sequence[str])
        token2idx (Dict[str, int])

    Returns:
        onehot (Sequence[int])

    """
    return [token2idx.get(w, token2idx[UNK_WORD]) for w in tokens]


if __name__ == "__main__":
    import pandas as pd
    smiles = pd.read_csv("smiles.csv")
    token2idx = build_vocabulary(smiles)
    idx2token = {i: token for token, i in token2idx.items()}
    vocab = [idx2token[i] for i in range(len(idx2token))]
    with open("alphabet_smiles.dat", "wt") as handle:
        handle.write("\n".join(vocab))
