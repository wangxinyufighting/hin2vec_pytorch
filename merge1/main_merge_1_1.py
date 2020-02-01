import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from walker import load_a_HIN_from_pandas_neighbors, getNeighbors
from merge1.model_merge_1 import NSTrainSet, HIN2vec, train

# set method parameters
window = 10
walk = 5
walk_length = 55
embed_size = 100
neg = 5
num_sample_nei = 5
sigmoid_reg = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

demo_edge = pd.read_csv('./train_tcm.csv', index_col=0)
edges = [demo_edge]
print('finish loading edges')

# init HIN
hin, edge_data_by_type, edge_types = load_a_HIN_from_pandas_neighbors(edges)
hin.window = window
dataset = NSTrainSet(hin.sample(walk_length, walk), hin.node_size, neg=neg)
neighbors = getNeighbors(
    edge_type_count=len(edge_types),
    num_nodes=hin.node_size,
    edge_data_by_type=edge_data_by_type,
    edge_types=edge_types,
    num_neighbor_samples=num_sample_nei)

hin2vec = HIN2vec(
    edge_types,
    neighbors,
    hin.node_size,
    num_sample_nei,
    hin.path_size,
    embed_size,
    sigmoid_reg)

# load model
# hin2vec.load_state_dict(torch.load('hin2vec.pt'))

# set training parameters
n_epoch = 1
batch_size = 64
log_interval = 200

if torch.cuda.is_available():
    print('Use ', device)
    hin2vec = hin2vec.to(device)
else:
    print('Use CPU')

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.AdamW(hin2vec.parameters())  # 原作者使用的是SGD？ 这里使用AdamW
loss_function = nn.BCELoss()

for epoch in range(n_epoch):
    train(neighbors, log_interval, hin2vec, device, data_loader, optimizer, loss_function, epoch)

torch.save(hin2vec, 'hin2vec.pt')

# set output parameters [the output file is a bit different from the original code.]
node_vec_fname = 'node_vec_merge_1_' + \
                 str(window) + '_' +\
                str(walk) + '_' +\
                str(walk_length) + '_' +\
                str(embed_size) + '_' +\
                str(neg) + '_' +\
                str(num_sample_nei) + '_' +\
                str(sigmoid_reg) + '_' +\
                str(n_epoch) + '_' +\
                str(batch_size) + '_' +\
                str(log_interval) +\
                 '.txt'
# path_vec_fname = 'meta_path_vec.txt'
path_vec_fname = None

print(f'saving node embedding vectors to {node_vec_fname}...')
node_embeds = pd.DataFrame(hin2vec.embeds_start.weight.data.numpy())
node_embeds.rename(hin.id2node).to_csv(node_vec_fname, sep=' ')

if path_vec_fname:
    print(f'saving meta path embedding vectors to {path_vec_fname}...')
    path_embeds = pd.DataFrame(hin2vec.embeds_path.weight.data.numpy())
    path_embeds.rename(hin.id2path).to_csv(path_vec_fname, sep=' ')

# save model
# torch.save(hin2vec.state_dict(), 'hin2vec.pt')