import os
import pickle
import numpy as np
import pandas as pd
import torch

data = pd.read_csv('data/twosides_ge_500.zip')
drug_smile_dict = {}

for id1, id2, smiles1, smiles2, relation in zip(data['Drug1_ID'], data['Drug2_ID'], data['Drug1'], data['Drug2'], data['New Y']):
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

drug_graph = read_pickle(os.path.join('data/preprocessed/twosides/drug_fig_data.pkl'))

def get_drug_fig(drug_smile_dict, drug_id):
    smiles = drug_smile_dict.get(drug_id)
    smiles_fig = torch.from_numpy(drug_graph[smiles])
    return smiles_fig