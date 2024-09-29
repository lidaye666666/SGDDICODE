import os
import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from torch_geometric.data.collate import collate
from data_preprocessing import CustomData
from test_representation_fig import get_drug_fig, get_drug_to_smiles
# %%
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class CustomBatch(Batch):
    @classmethod
    def from_data_list(cls, data_list, follow_batch = None, exclude_keys = None):
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=False,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch

class DrugDataset(Dataset):
    def __init__(self, data_df, drug_graph):
        self.data_df = data_df
        self.drug_graph = drug_graph
        self.drug_to_smiles = get_drug_to_smiles()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []
        head_fig = []
        tail_fig = []

        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_ID = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            h_graph = self.drug_graph.get(Drug1_ID)
            t_graph = self.drug_graph.get(Drug2_ID)
            n_graph = self.drug_graph.get(Neg_ID)

            pos_pair_h = h_graph
            pos_pair_t = t_graph
            neg_pair_h = n_graph
            neg_pair_t = t_graph

            head_fig.append(get_drug_fig(self.drug_to_smiles, Drug1_ID))
            tail_fig.append(get_drug_fig(self.drug_to_smiles, Drug2_ID))
            head_fig.append(get_drug_fig(self.drug_to_smiles, Neg_ID))
            tail_fig.append(get_drug_fig(self.drug_to_smiles, Drug2_ID))

            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)
            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)

            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))

        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index'])
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)

        return head_pairs, tail_pairs, rel, label,torch.stack(head_fig), torch.stack(tail_fig)

"""
这段代码定义了一个用于加载药物相互作用数据集的函数 load_ddi_dataset()。
该函数首先从指定的目录中加载药物图数据，然后加载分为训练集、验证集和测试集的数据文件，并将它们转换为 DrugDataset 对象。
接着，使用自定义的 DrugDataLoader 类将数据集包装成 PyTorch 的 DataLoader 对象，以便于训练和评估模型。
"""
class DrugDataLoader(DataLoader):
    """
    data: 数据集对象，即 DrugDataset 对象或其子类的实例。这个参数指定了要加载的数据集
    collate_fn=data.collate_fn: 数据集的整理函数，用于对每个批次的数据进行整理和处理
    这里直接使用了数据集对象的 collate_fn 方法，它会自动调用数据集对象的整理函数，以确保数据被正确地整理和处理
    """
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

"""
split_train_valid:用于将数据集分割为训练集和验证集
接收三个参数：
    data_df：包含样本信息的数据框（DataFrame）
    fold：用于随机分割的随机种子
    val_ratio：验证集的比例，默认为 0.2，表示验证集占整个数据集的比例
    

函数使用了 StratifiedShuffleSplit 来进行分层随机分割，这是一种常用的分层交叉验证方法
创建了一个 StratifiedShuffleSplit 对象 cv_split，它将数据集分成两个部分：训练集和验证集。n_splits 参数设为 2，表示只进行一次分割

cv_split.split() 方法对数据集进行分割，其中 X 参数是样本的索引（使用 range(len(data_df))），y 参数是样本的标签（使用 data_df['Y']）
next(iter(...)) 获取了迭代器的下一个元素，即一次分割的结果。这个结果包含了训练集和验证集的索引

根据分割结果，从原始数据框中提取了训练集和验证集的样本信息，并存储在 train_df 和 val_df 中
"""
def split_train_valid(data_df, fold, val_ratio=0.2):
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
        train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y = data_df['Y'])))

        train_df = data_df.iloc[train_index]
        val_df = data_df.iloc[val_index]

        return train_df, val_df

def load_ddi_dataset(root, batch_size, fold=0):
    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    train_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_train_fold{fold}.csv'))
    test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_test_fold{fold}.csv'))
    train_df, val_df = split_train_valid(train_df, fold=fold)

    train_set = DrugDataset(train_df, drug_graph)
    val_set = DrugDataset(val_df, drug_graph)
    test_set = DrugDataset(test_df, drug_graph)

    """
    train_set: 训练集数据，即一个 DrugDataset 对象，包含了训练集的样本信息和药物图数据
    batch_size: 批次大小，即每个批次中包含的样本数量
    shuffle=True: 表示在每个 epoch 开始时，是否对数据进行洗牌，以确保每个批次都是随机选择的样本
    num_workers=8: 表示用于加载数据的子进程数量。通过设置多个子进程来加载数据，可以加快数据加载速度，特别是当数据量较大时
    drop_last=True: 表示如果最后一个批次的样本数量不足一个批次大小，是否丢弃该批次。
        在训练过程中，通常会将最后一个批次不足一个批次大小的样本丢弃，以确保所有批次的样本数量都一致
    
    """

    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))
        
    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    train_loader, val_loader, test_loader = load_ddi_dataset(root='data/preprocessed/twosides', batch_size=256, fold=0)


# %%
