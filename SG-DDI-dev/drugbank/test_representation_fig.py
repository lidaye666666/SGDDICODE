import os
import pickle
import numpy as np
import pandas as pd
import torch

data = pd.read_csv('data/drugbank.tab', sep='\t')
drug_smile_dict = {}

for id1, id2, smiles1, smiles2, relation in zip(data['ID1'], data['ID2'], data['X1'], data['X2'], data['Y']):
    drug_smile_dict[id1] = smiles1
    drug_smile_dict[id2] = smiles2

def get_drug_to_smiles_sm1(Drug_ID):
    return drug_smile_dict.get(Drug_ID)

def get_drug_to_smiles():
    return drug_smile_dict

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

drug_graph = read_pickle(os.path.join('data/preprocessed/drugbank/drug_fig_data1024.pkl'))

def get_drug_fig(drug_smile_dict, drug_id):
    smiles = drug_smile_dict.get(drug_id)
    smiles_fig = torch.from_numpy(drug_graph[smiles])
    return smiles_fig