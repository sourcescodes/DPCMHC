import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
ramino = 'ARNDCQEGHILKMFPSTWYV'
amino = 'ARNDCQEGHILKMFPSTWYVX'
encoding_dict = {
    'A': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    'R': [1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
    'N': [1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    'D': [1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    'C': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
    'Q': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    'E': [1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    'G': [1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    'H': [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    'I': [1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    'L': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    'K': [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    'M': [0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    'F': [0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
    'P': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    'S': [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    'T': [0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
    'W': [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
    'Y': [0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
    'V': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    'X': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
}

def pep_encoding(seq):
    seq = seq[:11]
    encoded_matrix = np.zeros((11,10))
    for i, char in enumerate(seq):
        if char in encoding_dict and i < len(seq):
            encoded_matrix[i] = encoding_dict[char]
    return encoded_matrix
def mhc_encoding(seq):
    encoded_matrix = np.zeros((34, 10))
    for i, char in enumerate(seq):
        if char in encoding_dict and i < 34:
            encoded_matrix[i] = encoding_dict[char]
    return encoded_matrix

f = pd.read_csv('dataset/data_ba.csv')
pep = f['peptide'].values
mhc = f['mhc'].values
label = f['logic'].values
compound_pep=[]
compound_mhc=[]
compound_label=[]
for index in tqdm(range(len(pep))):
    temp_matrix = pep_encoding(pep[index])
    compound_pep.append(temp_matrix)

for index in tqdm(range(len(mhc))):
    temp_matrix = mhc_encoding(mhc[index])
    compound_mhc.append(temp_matrix)
for index in tqdm(range(len(label))):
    compound_label.append(label[index])


np.save('dataset/.npy', np.asarray(compound_pep).astype(float))
np.save('dataset/.npy', np.asarray(compound_mhc).astype(float))
np.save('dataset/.npy', np.asarray(compound_label).astype(float))



