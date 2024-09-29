import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GATConv
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import  softmax
from torch_scatter import scatter
from torch_geometric.utils import degree


class GlobalAttentionPool(nn.Module):

    def __init__(self, hidden_dim):
        # hidden_dim:代表输入特征的维度:64
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)
        # GraphConv:用于图卷积操作的神经网络层
        # 接受输入特征维度hidden_dim和输出特征维度为1

    def forward(self, x, edge_index, batch):
        """
        前向传播函数
        接受三个参数：
        x: 输入特征张量，代表节点特征
        edge_index: 图的边索引，描述了节点之间的连接关系
        batch: 一个张量，用于描述图中的不同子图
        """
        x_conv = self.conv(x, edge_index)
        # 在前向传播中，首先将输入特征x通过self.conv层进行图卷积操作，得到x_conv，这个操作将节点特征更新为新的特征表示
        scores = softmax(x_conv, batch, dim=0)
        # 接下来，对x_conv进行 softmax 操作，以计算每个节点的权重分数。
        # softmax函数的第二个参数batch用于指定每个节点所属的子图，而dim=0表示在每个子图中进行 softmax 操作。
        gx = global_add_pool(x * scores, batch)
        # 使用全局加池化（global_add_pool）函数，将节点特征乘以对应的分数，
        # 然后对每个子图中的节点进行求和，最终得到一个全局的特征向量gx，其中包含了每个子图的全局信息。

        return gx


class DMPNN(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        """
        edge_dim: 边特征的维度
        n_feats: 节点特征的维度
        n_iter: 迭代的次数，用于DMPNN中的迭代次数
        """

        # print("edge_dim # 6")
        # print(edge_dim)
        # print("n_feats # 64")
        # print(n_feats)
        # print("n_iter # 10")
        # print(n_iter)

        super().__init__()
        self.n_iter = n_iter
        # 将传入的n_iter参数存储为对象的成员变量，表示模型的迭代次数
        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        # 创建一个线性层self.lin_u，用于将节点特征data.x的维度映射到相同的维度，没有偏置
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        # 创建另一个线性层self.lin_v，用于将节点特征data.x的维度映射到相同的维度，也没有偏置
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)
        # 创建第三个线性层self.lin_edge，将边特征data.edge_attr的维度映射到节点特征维度，同样没有偏置
        self.att = GlobalAttentionPool(n_feats)
        # 创建一个名为self.att的全局池化层，用于实现全局池化操作，汇总图的全局信息
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        # 创建一个可学习的参数self.a，它是一个1xn_featsxn_iter的张量，将在训练过程中学习
        self.lin_gout = nn.Linear(n_feats, n_feats)
        # 创建一个线性层self.lin_gout，用于将节点特征映射到相同的维度
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))
        # 创建另一个可学习的参数self.a_bias，它是一个1x1xn_iter的张量，也将在训练中学习

        glorot(self.a)
        # 使用glorot函数对参数self.a进行初始化，这是一种常用的权重初始化方法
        self.lin_block = LinearBlock(n_feats)
        # 创建一个名为self.lin_block的线性块，用于对节点特征进行进一步处理
    def forward(self, data):
        # 接受一个名为data的输入参数，通常包含了图数据
        edge_index = data.edge_index
        # print("data.x")
        # print(data.x.shape) # 13904,64
        # print("data.edge_index_batch")
        # print(data.edge_index_batch.shape) # torch.Size(2,29714)
        # print("data.x")
        # print(data.edge_index.shape)  # torch.Size(2,29714)

        # 从输入数据data中提取边索引，表示图的边连接关系
        # 因此我们应该在开始时为每个键分配一个键级特征向量（即论文中的 h_{ij}^{(0)}）
        # Recall that we have converted the node graph to the line graph, 
        # so we should assign each bond a bond-level feature vector at the beginning (i.e., h_{ij}^{(0)}) in the paper).
        edge_u = self.lin_u(data.x)
        # 使用self.lin_u层将输入节点特征data.x映射到相同的维度，得到edge_u
        edge_v = self.lin_v(data.x)
        # 使用self.lin_v层将输入节点特征data.x映射到相同的维度，得到edge_v
        edge_uv = self.lin_edge(data.edge_attr)
        # 使用self.lin_edge层将边特征data.edge_attr映射到节点特征维度，得到edge_uv
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        # 计算边属性edge_attr，这是节点特征和边特征的组合，通过平均得到
        out = edge_attr
        # 初始化out为edge_attr，表示初始的边级特征


        # 下面的代码展示了图卷积和子结构注意力
        # The codes below show the graph convolution and substructure attention.
        out_list = []
        gout_list = []
        # 创建了两个空列表out_list和gout_list，用于存储每轮迭代中的中间结果
        # out_list存储图卷积的输出
        # gout_list存储子结构注意力的输出
        for n in range(self.n_iter):
            # 这是一个循环，迭代次数由self.n_iter指定，通常用于多轮的图卷积操作
            # Lines 61 and 62 are the main steps of graph convolution.
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            """
            这一行执行了图卷积的主要步骤
            scatter函数用于根据data.line_graph_edge_index将节点级特征out聚合到线图的边级特征
            data.line_graph_edge_index[0]表示线图的源节点
            data.line_graph_edge_index[1]表示线图的目标节点  
            dim_size=edge_attr.size(0)用于指定聚合的维度大小，通常是边的数量
            reduce='add'表示采用求和的方式进行聚合，以计算邻居节点的总和
            """
            out = edge_attr + out
            # 这一行将图卷积的输出out与边属性edge_attr相加，实现了文献中的"Equation (1)"
            # Equation (1) in the paper
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            # 这一行使用全局注意力层self.att来计算子结构注意力。它将图卷积的输出out
            # 线图的边索引data.line_graph_edge_index以及批次信息data.edge_index_batch作为输入，并计算子结构注意力
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))
            # 这两行将图卷积的输出out和子结构注意力的输出添加到相应的列表out_list和gout_list中，以便后续的堆叠和计算。
        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        # 这两行将gout_list和out_list中的结果在最后一个维度上进行堆叠，
        # 得到gout_all和out_all，分别表示子结构注意力的输出和图卷积的输出
        # Substructure attention, Equation (3)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        # 这一行计算子结构注意力的得分。它首先将子结构注意力输出gout_all与可学习参数self.a相乘，
        # 然后在第一个维度上求和，并加上可学习参数self.a_bias

        # Substructure attention, Equation (4),
        # Suppose batch_size=64 and iteraction_numbers=10. 
        # Then the scores will have a shape of (64, 1, 10), 
        # which means that each graph has 10 scores where the n-th score represents the importance of substructure with radius n.
        scores = torch.softmax(scores, dim=-1)
        # We should spread each score to every line in the line graph.
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)
        # Weighted sum of bond-level hidden features across all steps, Equation (5).
        out = (out_all * scores).sum(-1)
        # Return to node-level hidden features, Equations (6)-(7).
        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x


# 对输入特征进行多层线性变换和非线性激活
class LinearBlock(nn.Module):

    def __init__(self, n_feats):
        # 接受一个参数n_feats，表示输入特征的维度
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        # 创建一个成员变量self.snd_n_feats，它是输入特征维度n_feats的6倍，用于定义中间线性层的维度
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.LeakyReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.LeakyReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.LeakyReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.LeakyReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        # 接受一个名为x的输入参数，通常是特征向量
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        # 将它们的输出与输入相加并除以2，以实现残差连接
        x = (self.lin4(x) + x) / 2
        # 应用残差连接
        x = self.lin5(x)
        # 将特征通过最后一轮的线性变换self.lin5，将特征维度映射回原始的n_feats
        return x   
        # 返回处理后的特征向量x，这个特征向量经过多轮线性变换和非线性激活，用于编码输入数据的信息。通常，这种模块用于在深度神经网络中构建复杂的特征表示

class DrugEncoder(torch.nn.Module):
    # 将输入数据（通常是代表化学分子的图数据）编码成特征向量
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        """
        in_dim: 输入特征的维度，通常表示节点特征的维度  70
        edge_in_dim: 输入边特征的维度，通常表示边特征的维度
        hidden_dim: 隐藏层的维度，用于中间特征表示，缺省值为64
        n_iter: 迭代的次数，用于DMPNN中的迭代次数，缺省值为10
        """
        super().__init__()
        """
        self.mlp是一个包含多个线性层、激活函数和批量归一化的神经网络序列。这个MLP用于对输入特征进行前处理。MLP的结构如下
        """
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # 第一层：线性层（nn.Linear），将输入特征维度in_dim映射到hidden_dim，没有偏置
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # 第二层：线性层，将hidden_dim映射到hidden_dim
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # 第三层：线性层，将hidden_dim映射到hidden_dim
            nn.BatchNorm1d(hidden_dim), 
        )
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        # 创建一个线性层self.lin0，它将输入特征维度in_dim映射到hidden_dim，用于进一步处理输入数据
        self.line_graph = DMPNN(edge_in_dim, hidden_dim, n_iter)
        # 接受边特征维度edge_in_dim、隐藏维度hidden_dim和迭代次数n_iter作为参数

    def forward(self, data):
        # 当调用模块时将执行的计算过程。接受一个名为data的输入参数，通常包含了图数据
        data.x = self.mlp(data.x)
        # 将输入数据data.x通过MLP模型self.mlp进行前处理，将输入特征映射到更高维度的特征表示
        x = self.line_graph(data)
        # 将前处理后的数据data传递给self.line_graph，这是DMPNN模型，用于对图数据进行进一步编码，得到特征向量x
        return x
        # 将编码后的特征向量x作为输出返回，这个特征向量可以用于后续任务，例如药物属性预测或分子相似性计算。

        # DrugEncoder是一个用于将化学分子的图数据编码成特征向量的神经网络模块，它包括前处理MLP和DMPNN编码器，用于学习分子的表示

class SG_DDI(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        super(SG_DDI, self).__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter)
        self.h_gpool = GlobalAttentionPool(hidden_dim)
        self.t_gpool = GlobalAttentionPool(hidden_dim)
        self.lin = nn.Sequential(
            # (128,128)
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(),
            # (128,64)
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # figuraprint
        self.hidden_size = 64
        self.lin_fig = nn.Sequential(
            nn.Linear(1705, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
        )
        # (86,64)
        self.rmodule = nn.Embedding(86, hidden_dim)
        # (64,64)
        self.w_j = nn.Linear(hidden_dim, hidden_dim)
        self.w_i = nn.Linear(hidden_dim, hidden_dim)
        # (64,64)
        self.prj_j = nn.Linear(hidden_dim, hidden_dim)
        self.prj_i = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, triples,head_fig,tail_fig):
        h_data, t_data, rels = triples

        # figuraprint
        h_final = self.lin_fig(head_fig.float())
        t_final = self.lin_fig(tail_fig.float())
        pair_conv = torch.cat([h_final, t_final], dim=1)
        # print("pair_conv")
        # print(pair_conv.shape)
        # pair_conv
        # torch.Size([512, 128])
        x_h = self.drug_encoder(h_data)
        x_t = self.drug_encoder(t_data)

        g_h = self.h_gpool(x_h, h_data.edge_index, h_data.batch)
        g_t = self.t_gpool(x_t, t_data.edge_index, t_data.batch)

        g_h_align = g_h.repeat_interleave(degree(t_data.batch, dtype=t_data.batch.dtype), dim=0)
        g_t_align = g_t.repeat_interleave(degree(h_data.batch, dtype=h_data.batch.dtype), dim=0)


        h_scores = (self.w_i(x_h) * self.prj_i(g_t_align)).sum(-1)
        h_scores = softmax(h_scores, h_data.batch, dim=0)

        t_scores = (self.w_j(x_t) * self.prj_j(g_h_align)).sum(-1)
        t_scores = softmax(t_scores, t_data.batch, dim=0)

        h_final = global_add_pool(x_h * g_t_align * h_scores.unsqueeze(-1), h_data.batch)
        t_final = global_add_pool(x_t * g_h_align * t_scores.unsqueeze(-1), t_data.batch)


        pair = torch.cat([h_final, t_final], dim=-1)

        pair = torch.cat([pair_conv, pair], dim=-1)

        rfeat = self.rmodule(rels)
        logit = (self.lin(pair) * rfeat).sum(-1)

        return logit