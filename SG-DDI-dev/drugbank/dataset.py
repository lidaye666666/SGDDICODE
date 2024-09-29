import os
import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from data_preprocessing import CustomData
from test_representation_fig import get_drug_fig, get_drug_to_smiles

# %%
"""
filename 是要读取的 pickle 文件的路径
open(filename, 'rb') 打开文件，使用二进制模式读取（'rb' 表示以只读二进制模式打开文件
pickle.load(f) 使用 pickle 模块的 load() 函数从文件中加载对象
返回从 pickle 文件中加载的对象
"""
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class DrugDataset(Dataset):
    """
    data_df，包含了数据集中的样本信息的 DataFrame
    drug_graph，一个字典，包含了药物图（drug graph）的信息
    """
    def __init__(self, data_df, drug_graph):
        self.data_df = data_df
        self.drug_graph = drug_graph
        self.drug_to_smiles = get_drug_to_smiles()
    # 返回数据集中样本的数量

    def __len__(self):
        return len(self.data_df)

    # 根据给定的索引 index 返回对应的数据样本

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    """
    collate_fn 方法中，它根据批次中的每个样本，从药物图中获取正样本和负样本的相关信息，并将它们放入列表中。
    然后，它将这些列表转换为 PyTorch 的张量，并返回两个药物的特征（head_pairs 和 tail_pairs）、关系标签（rel）以及样本标签（label）。
    """
    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []
        head_fig = []
        tail_fig = []

        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']

            # 这行代码将字符串 Neg_samples 根据 '$' 符号进行拆分，然后将拆分得到的结果分别赋值给变量 Neg_ID 和 Ntype
            # Neg_ID 是负样本的 ID
            # Ntype 是负样本的类型

            Neg_ID, Ntype = Neg_samples.split('$')
            h_graph = self.drug_graph.get(Drug1_ID)
            t_graph = self.drug_graph.get(Drug2_ID)
            n_graph = self.drug_graph.get(Neg_ID)

            pos_pair_h = h_graph
            pos_pair_t = t_graph

            if Ntype == 'h':
                neg_pair_h = n_graph
                neg_pair_t = t_graph
            else:
                neg_pair_h = h_graph
                neg_pair_t = n_graph

            head_fig.append(get_drug_fig(self.drug_to_smiles, Drug1_ID))
            tail_fig.append(get_drug_fig(self.drug_to_smiles, Drug2_ID))
            head_fig.append(get_drug_fig(self.drug_to_smiles, Neg_ID))
            tail_fig.append(get_drug_fig(self.drug_to_smiles, Drug2_ID))

            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)
            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)

            # 每个正样本和负样本都有相同的关系标签，所以这里使用了相同的关系标签张量。
            # 这样做是为了确保在构建数据集时每个样本的关系标签都被正确地处理和添加。
            # torch.LongTensor([Y]) 创建的张量类型是长整型（LongTensor），这意味着它存储的值是整数，并且通常用于表示类别或标签。
            # 因此，这些张量存储了样本的关系标签，用于表示样本之间的相互作用情况。
            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            # 这两行代码向 label_list 中添加了两个张量对象。
            # 这里 torch.FloatTensor([1]) 和 torch.FloatTensor([0])
            # 创建了包含单个元素 1 和 0 的浮点型张量，分别代表正样本和负样本的标签。
            # 在许多二分类任务中，通常使用 1 表示正样本，0 表示负样本。
            # 这里的代码也遵循了这个约定。通过将这些标签添加到 label_list 中，确保了在构建数据集时每个样本的标签都被正确地处理和添加。
            # 这些标签在训练模型时通常被用作损失函数的目标值

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))
        """
        使用 Batch.from_data_list() 函数将 head_list 和 tail_list 转换为 PyTorch Geometric 库中的 Batch 对象。
        这个函数将一组数据列表转换为一个包含多个图数据的批次（Batch），这些图数据可以方便地用于神经网络模型的训练和评估
        
        head_list 和 tail_list 包含了每个样本的头部和尾部药物的特征数据，这些数据可能是图数据的形式。
        通过 Batch.from_data_list() 函数，将这些图数据转换为一个 Batch 对象，
        其中包含了多个图数据，并且可以指定要跟随的属性（例如 'edge_index'），以便在将图数据传递给神经网络模型时一起传递
        """
        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index'])

        # torch.cat() 函数将 rel_list 中的张量沿着指定的维度进行拼接，生成一个新的张量 rel
        # rel_list 包含了每个样本的关系标签，每个关系标签是一个单独的张量对象
        # 通过 torch.cat(rel_list, dim=0)，将这些张量沿着维度0进行拼接，即将它们堆叠在一起，形成一个新的张量
        # 将多个样本的标签数据合并成一个张量，以便于在模型中进行批处理训练或评估。拼接完成后，得到的 rel 张量将包含整个数据集中所有样本的关系标签
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)

        return head_pairs, tail_pairs, rel, label, torch.stack(head_fig), torch.stack(tail_fig)

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


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
    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))
        
    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    train_loader, val_loader, test_loader = load_ddi_dataset(root='data/preprocessed/drugbank', batch_size=256, fold=0)


# %%
