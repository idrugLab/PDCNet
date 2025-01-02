import numpy as np

def zscale_padding(seq_list, padding):

    zscale = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30],
        'C': [0.84, -1.67, 3.71, 0.18, -2.65],
        'D': [3.98, 0.93, 1.93, -2.46, 0.75],
        'E': [3.11, 0.26, -0.11, -0.34, -0.25],
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62],
        'G': [2.05, -4.06, 0.36, -0.82, -0.38],
        'H': [2.47, 1.95, 0.26, 3.90, 0.09],
        'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
        'K': [2.29, 0.89, -2.49, 1.49, 0.31],
        'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98],
        'N': [3.05, 1.62, 1.04, -1.15, 1.61],
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00],
        'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
        'R': [3.52, 2.50, -3.50, 1.99, -0.17],
        'S': [2.39, -1.07, 1.15, -1.39, 0.67],
        'T': [0.75, -2.18, -1.12, -1.46, -0.40],
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59],
        'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
        '-': [0.00, 0.00, 0.00, 0.00, 0.00]  # For gaps or unknown amino acids
    }
    feat_list = []
    for seq in seq_list:
        feat = [zscale.get(aa, zscale['-']) for aa in seq]
        feat += [[0] * 5] * (padding - len(seq))
        feat_list.append(feat)
    return np.array(feat_list)


def one_hot_padding(seq_list, padding):
    """
    Generate features for aa sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences,
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0] * 20
        one_hot[aa[i]][i] = 1
    for seq in seq_list:
        feat = [one_hot[aa] for aa in seq] + [[0] * 20] * (padding - len(seq))
        feat_list.append(feat)
    return np.array(feat_list)


def position_encoding(padding, d=20, b=1000):
    """
    Position encoding features introduced in "Attention is All You Need",
    the base (b) is changed to 1000 for the short length of peptides.
    """
    pos_encoding = np.zeros((padding, d))
    for pos in range(padding):
        for i in range(d // 2):
            pos_encoding[pos, 2 * i] = np.sin(pos / (b ** (2 * i / d)))
            pos_encoding[pos, 2 * i + 1] = np.cos(pos / (b ** (2 * i / d)))
    return pos_encoding


# BLOSUM62 matrix
blosum62 = {
    'A': [4, -1, -2, -2, 0, -2, -1, 0, -2, -1, -1, -1, -1, -1, -1, 1, 0, 0, -3, -2],
    'C': [-1, 9, -3, -4, -2, -3, -3, -3, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    'D': [-2, -3, 6, 2, -3, 1, -1, -1, -1, -4, -3, 1, 0, -3, -3, 0, -1, -3, -4, -3],
    'E': [-2, -4, 2, 5, -3, 0, -2, -1, 1, -3, -2, 0, 0, -3, -2, 0, -1, -2, -3, -2],
    'F': [0, -2, -3, -3, 6, -3, -1, -3, -1, 0, 0, -3, -4, -3, -3, -2, -2, -3, 1, 3],
    'G': [-2, -3, 1, 0, -3, 6, -2, -4, -2, -4, -3, 0, -2, -3, -2, 0, -2, -3, -2, -3],
    'H': [-1, -3, -1, -2, -1, -2, 8, -3, -1, -3, -2, -1, -2, -1, -2, -2, -2, -2, 2, 2],
    'I': [0, -3, -1, -1, -3, -4, -3, 4, -3, -3, -3, -1, -3, -3, -3, -3, -1, 2, -3, -1],
    'K': [-2, -3, -1, 1, -1, -2, -1, -3, 5, -2, -1, 0, -1, 1, 0, -1, 0, -2, -3, -2],
    'L': [-1, -1, -4, -3, 0, -4, -3, -3, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1],
    'M': [-1, -1, -3, -2, 0, -3, -2, -3, -1, 2, 5, -2, -2, 0, -1, -2, -1, 1, -1, -1],
    'N': [-1, -3, 1, 0, -3, 0, -1, -1, 0, -3, -2, 6, 2, -3, -3, 1, 0, -3, -4, -2],
    'P': [-1, -3, 0, 0, -4, -2, -2, -3, -1, -3, -2, 2, 7, -1, -1, -1, -1, -2, -4, -3],
    'Q': [-1, -3, -3, -3, -3, -3, -1, -3, 1, -2, 0, -3, -1, 5, 2, -1, 0, -2, -2, -1],
    'R': [-1, -3, -3, -2, -3, -2, -2, -3, 0, -2, -1, -3, -1, 2, 5, -1, 0, -2, -3, -2],
    'S': [1, -1, 0, 0, -2, 0, -2, -3, -1, -2, -2, 1, -1, -1, -1, 4, 1, -2, -3, -2],
    'T': [0, -1, -1, -1, -2, -2, -2, -1, 0, -1, -1, 0, -1, 0, 0, 1, 5, 0, -2, -2],
    'V': [0, -1, -3, -2, -3, -3, -2, 2, -2, 1, 1, -3, -2, -2, -2, -2, 0, 4, -3, -1],
    'W': [-3, -2, -4, -3, 1, -2, 2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2],
    'Y': [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # For unknown amino acids
}

def blosum62_padding(seq_list, padding):
    """
    Generate BLOSUM62 features for aa sequences with zero padding.
    Input: seq_list: list of sequences,
           padding: padding length, >= max sequence length.
    Output: BLOSUM62 encoding of sequences.
    """
    feat_list = []
    for seq in seq_list:
        feat = [blosum62.get(aa, blosum62['X']) for aa in seq] + [[0] * 20] * (padding - len(seq))
        feat_list.append(feat)
    return np.array(feat_list)

def combined_encoding(seq_list, padding):
    """
    Combine one-hot encoding, position encoding, and BLOSUM62 encoding.
    Input: seq_list: list of sequences,
           padding: padding length, >= max sequence length.
    Output: combined encoding of sequences.
    """
    one_hot_encoded = one_hot_padding(seq_list, padding)
    pos_encoded = position_encoding(padding)
    blosum_encoded = blosum62_padding(seq_list, padding)
    zscale_encoded = zscale_padding(seq_list, padding)

    pos_encoded_expanded = np.expand_dims(pos_encoded, axis=0).repeat(len(seq_list), axis=0)


    combined_encoding = np.concatenate(
        (one_hot_encoded, pos_encoded_expanded, blosum_encoded, zscale_encoded), axis=2
    )
    return combined_encoding