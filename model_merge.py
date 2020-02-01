import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class HIN2vec(nn.Module):
    def __init__(self, edge_types, neighbors, node_size, num_sample_nei, path_size, embed_dim, sigmoid_reg=True, r=False):
        super().__init__()
        def binary_reg(x: torch.Tensor):
            raise NotImplementedError()

        self.reg = torch.sigmoid if sigmoid_reg else binary_reg
        self.__initialize_model(node_size, path_size, embed_dim, r)
        self.neighbors = neighbors
        self.edge_types = edge_types
        self.trans1 = nn.Linear(embed_dim, embed_dim)
        self.trans2 = nn.Linear(embed_dim * num_sample_nei, embed_dim)

    def __initialize_model(self, node_size, path_size, embed_dim, r):
        self.embeds_start = nn.Embedding(node_size, embed_dim)
        self.embeds_end = self.embeds_start if r else nn.Embedding(node_size, embed_dim)
        self.embeds_path = nn.Embedding(path_size, embed_dim)

    def getNeighborNodes(self, neighbors, nodes):
        '''
        :param neighbors: [node_1][edge_type] = [node2,....]
        '''
        node_neighbors = {}
        for i in nodes:
            temp = []
            for n in neighbors[i][:]:
                temp.extend(n)
            node_neighbors[i] = temp

        return node_neighbors

    def forward(self, neighbors, start_node, end_node, path):
        embeds_start_nodes = []
        for n in start_node:
            embeds_neighbors_all_type = []
            for e in range(len(self.edge_types)):
                node_neighbors_by_type = torch.tensor(neighbors[n][e])
                embeds_neighbors_by_type = self.embeds_start(node_neighbors_by_type)
                # aggregator
                embeds_neighbors_by_type_agg = self.trans1(torch.mean(embeds_neighbors_by_type, dim=0))
                embeds_neighbors_all_type.append(embeds_neighbors_by_type_agg)
            embeds_start_nodes.append(torch.cat([i for i in embeds_neighbors_all_type]))
        embeds_start_nodes = torch.stack(embeds_start_nodes, dim=0)
        # embeds_start_nodes = self.embeds_start(start_node)
        embeds_start_nodes = self.trans2(embeds_start_nodes)
        embeds_end_ndoes = self.embeds_end(end_node)
        embeds_path = self.embeds_path(path)
        embeds_path = self.reg(embeds_path)

        agg = torch.mul(embeds_start_nodes, embeds_end_ndoes)
        agg = torch.mul(agg, embeds_path)

        output = torch.sigmoid(torch.sum(agg, axis=1))

        return output

    # def forward(self, neighbors, start_node, end_node, path):
    #     node_neighbors = self.getNeighborNodes(neighbors, start_node)
    #     embeds_start_nodes = []
    #     for node, node_neigh in node_neighbors.items():
    #         node_neigh = torch.tensor(node_neigh)
    #         embeds_start_nodes.append(torch.cat([i for i in self.embeds_start(node_neigh)]))
    #     embeds_start_nodes = torch.stack(embeds_start_nodes, dim=0)
    #     # embeds_start_nodes = self.embeds_start(start_node)
    #     embeds_start_nodes = self.trans(embeds_start_nodes)
    #     embeds_end_ndoes = self.embeds_end(end_node)
    #     embeds_path = self.embeds_path(path)
    #     embeds_path = self.reg(embeds_path)
    #
    #     agg = torch.mul(embeds_start_nodes, embeds_end_ndoes)
    #     agg = torch.mul(agg, embeds_path)
    #
    #     output = torch.sigmoid(torch.sum(agg, axis=1))
    #
    #     return output

def train(neighbors, log_interval, model, device, train_loader: DataLoader, optimizer, loss_function, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(neighbors, data[:, 0], data[:, 1], data[:, 2])
        loss = loss_function(output.view(-1), target)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0:
            print(f'\rTrain Epoch: {epoch} '
                  f'[{idx * len(data)}/{len(train_loader.dataset)} ({100. * idx / len(train_loader):.3f}%)]\t'
                  f'Loss: {loss.item():.3f}\t\t',
                  # f'data = {data}\t target = {target}',
                  end='')

class NSTrainSet(Dataset):
    """
    完全随机的负采样 todo 改进一下？
    """

    def __init__(self, sample, node_size, neg=5):
        """
        :param node_size: 节点数目
        :param neg: 负采样数目
        :param sample: HIN.sample()返回值，(start_node, end_node, path_id)
        """

        print('init training dataset...')

        l = len(sample)

        x = np.tile(sample, (neg + 1, 1))
        y = np.zeros(l * (1 + neg))
        y[:l] = 1

        # x[l:, 2] = np.random.randint(0, path_size - 1, (l * neg,))
        x[l:, 1] = np.random.randint(0, node_size - 1, (l * neg,))

        self.x = torch.LongTensor(x)
        self.y = torch.FloatTensor(y)
        self.length = len(x)

        print('finished')

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length