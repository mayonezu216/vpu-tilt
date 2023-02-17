import numpy as np
import torch
import os
import shutil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_dist(embedding):
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import MinMaxScaler
    #metric = ["cosine", "euclidean", "cityblock"]
    dist = pairwise_distances(embedding)
    mm = MinMaxScaler()
    dist = mm.fit_transform(dist)
    dist = np.exp(-dist)
    return dist

def get_cos_adj(embedding): # not a strict distant, we can also directly use embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    adj = cosine_similarity(embedding)
    return adj

def get_euclidean_adj(embedding):
    from sklearn.metrics.pairwise import euclidean_distances
    adj =  euclidean_distances(embedding)
    return adj

def get_Haversine_adj(embedding):
    '''
    The Haversine (or great circle) distance is the angular distance between two points on the surface of a sphere.
    The first coordinate of each point is assumed to be the latitude, the second is the longitude, given in radians.
    The dimension of the data must be 2
    '''
    from sklearn.metrics.pairwise import haversine_distances
    adj = haversine_distances(embedding)
    return adj


def constructing_graph(val_loader, model):
    model.eval()
    total_embeddings = []
    total_pred = []
    for i, data in enumerate(val_loader):
        images, labels = data
        images = images.to(device)
        outputs, encoded = model(images)
        score, pred = torch.max(outputs, 1)
        pred = pred.detach().cpu().numpy()
        encoded_features = encoded.detach().cpu().numpy()
        total_pred.append(pred)
        total_embeddings.append(encoded_features)
    total_pred = np.squeeze(np.row_stack(total_pred))
    print('pred shape',len(total_pred))
    total_embeddings = np.row_stack(total_embeddings)

    return total_embeddings,total_pred

def create_temp_dir(root):
    g = os.walk(root)
    for path, dir_list, file_list in g:
        print(dir_list)
        for file_name in file_list:
            name = os.path.join(path, file_name)
            if (any(chr.isdigit() for chr in name)):
                if (len(name)<80):
                    slide_name = name.split('-')[1].split('_')[0]
                    if (name.split('/')[-2].isdigit() == 0):
                        if not os.path.exists(os.path.join(root+'/'+ slide_name)):
                            os.makedirs(os.path.join(root+'/' + slide_name+'/' + name.split('/')[-2]))
                    shutil.copy(path+'/'+file_name, os.path.join(root+'/' + slide_name +'/' + name.split('/')[-2]))
    print('Finish creating graph dir.')
    return True
def del_temp_slide_dir(root):
    filenames = os.listdir(root)
    for name in filenames:
        if name.isdigit():
            shutil.rmtree(os.path.join(root, name))
    return True

def del_temp_graph_dir(root):
    filenames = os.listdir(root)
    for name in filenames:
        if name == 'raw' or name == 'processed'or name == 'saved_graph':
            shutil.rmtree(os.path.join(root, name))
        elif name[-3:]=='npy':
            os.remove(os.path.join(root, name))
    return True


'''

from scipy import sparse
adj = np.random.rand(4,4)
adj = np.triu(adj)
adj += adj.T - np.diag(adj.diagonal())
print(adj)
temp = (adj>0.5)
adj = adj*temp
print(adj)
adj = sparse.csr_matrix(adj).tocoo()
print(adj)
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)
print(edge_index)
edge_feat  = adj.data
# for i in adj:
#     edge_feat.append(adj)
print(edge_feat.shape)
'''
# def get_adj(embedding):
#       #want to use the anchor information
#     n_clusters = embedding.shape[0]
#     for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
#         avg_dist = np.zeros((n_clusters, n_clusters))
#             plt.figure(figsize=(5, 4.5))
#             for i in range(n_clusters):
#                 for j in range(n_clusters):
#                     avg_dist[i, j] = pairwise_distances(
#                         X[y == i], X[y == j], metric=metric
#                     ).mean()
#             avg_dist /= avg_dist.max()

# import networkx as nx
# from sklearn.preprocessing import StandardScaler
# import torch
# # load graph from networkx library
# G = nx.karate_club_graph()
#
# # retrieve the labels for each node
# labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64)
#
# # create edge index from
# adj = nx.to_scipy_sparse_matrix(G).tocoo()
# print(adj)
# row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
# col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
# edge_index = torch.stack([row, col], dim=0)
# print(edge_index)
# # using degree as embedding
# embeddings = np.array(list(dict(G.degree()).values()))
# print('embeddings shape', embeddings.shape)
# # normalizing degree values
# scale = StandardScaler()
# embeddings = scale.fit_transform(embeddings.reshape(-1,1))
#
import os.path as osp
import torch
import glob
import numpy as np
from torch_geometric.data import Dataset, Data, download_url
from tqdm import tqdm
from scipy import sparse
from graph_construction import get_cos_adj,get_dist,get_euclidean_adj,get_Haversine_adj

class UrineDataset_for_graph(Dataset):
    def __init__(self, root = '.', transform=None, pre_transform=None):
        '''
        root: where the dataset should be stored. This folder is split into raw_dir (download dataset)
        and processed_dir (processed data).
        '''
        # self.embeddings = embeddings
        real_root_list = root.split('/')[:-1]
        self.real_root = '/'.join(real_root_list)
        self.filenames = glob.glob(root + '/*' + '/BD*')
        self.graph_list = []
        # print(self.real_root )
        # print(self.filenames )
        super(UrineDataset_for_graph,self).__init__(root,transform, pre_transform)

    @property
    def raw_file_names(self):
        '''If this file exists in raw_dir, the download function is not implemented.
        '''
        return '1.txt'

    @property
    def processed_file_names(self):
        return 'not_implemented.pt'

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        if not os.path.exists(os.path.join(self.real_root,'saved_graph')):
            os.makedirs(os.path.join(self.real_root,'saved_graph'))
        for name in self.filenames:
            if (all(chr.isdigit() for chr in name.split('-')[-1].split('.')[0])):
                self.graph_list.append(name)
            for name in self.graph_list:
                embeddings = np.load(name)
                node_feats = self._get_node_features(embeddings)
                edge_index = self._get_adjacency_info(embeddings)
                edge_feats = self._get_edge_features(embeddings)

            if name.split('/')[-2]=='cancer':
                self.label = 1
            if name.split('/')[-2]=='benign':
                self.label = 0
            if name.split('/')[-2]=='atypical':
                self.label = 0
            if name.split('/')[-2]=='suspicious':
                self.label = 1

            data = Data(x = node_feats,
                        edge_index = edge_index,
                        edge_attr = edge_feats,
                        y = self.label
                        )
            # os.makedirs(os.path.join(self.real_root,'saved_graph'))
            processed_dir = os.path.join(self.real_root,'saved_graph')
            save_name = name.split('.')[0].split('/')[-1]
            torch.save(data,osp.join(processed_dir,
                                 f'graph_{save_name}.pt'))


        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)



    def _get_node_features(self,embeddings):
        return torch.tensor(embeddings,dtype = torch.float)

    def _get_edge_features(self,embeddings):
        adj_type = 'dist'
        if adj_type == 'euclidean':
            adj = get_euclidean_adj(embeddings)
        elif adj_type == 'Haversine':
            adj = get_Haversine_adj(embeddings)
        elif adj_type == 'dist':
            adj = get_dist(embeddings)
        else:
            adj = get_cos_adj(embeddings)
        # adj = np.random.rand(4, 4)
        # adj = np.triu(adj)
        # adj += adj.T - np.diag(adj.diagonal())
        # print(adj)
        temp = (adj > 0.9)
        adj = adj * temp
        # print(adj)
        adj = sparse.csr_matrix(adj).tocoo()
        edge_feat = adj.data
        # print(torch.tensor(edge_feat,dtype = torch.long).size())
        # edge_feat = np.reshape(adj.data,[adj.data.shape,1])
        return torch.tensor(edge_feat,dtype = torch.long)


    def _get_adjacency_info(self,embeddings):
        adj_type = 'dist'
        if adj_type == 'euclidean':
            adj = get_euclidean_adj(embeddings)
        elif adj_type == 'Haversine':
            adj = get_Haversine_adj(embeddings)
        elif adj_type == 'dist':
            adj = get_dist(embeddings)
        else:
            adj = get_cos_adj(embeddings)

        temp = (adj > 0.9)
        adj = adj * temp
        # print(adj)
        adj = sparse.csr_matrix(adj).tocoo()
        # print(adj)
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        # print(edge_index)
        # return torch.tensor(edge_index,dtype = torch.long).clone().detach()

        return torch.as_tensor(edge_index, dtype=torch.long)
    # def _get_labels(self,embeddings):

    def len(self):
        # return len(self.processed_file_names)
        return len(self.filenames)



    def get(self, idx):
        processed_dir = os.path.join(self.real_root, 'saved_graph')
        graph_name = self.graph_list[idx].split('.')[0].split('/')[-1]
        data = torch.load(osp.join(processed_dir, f'graph_{graph_name}.pt'))
        return data


# embeddings = np.random.rand(30,2048)
# dataset = UrineDataset(root = 'data/')
# print(dataset[0].edge_index.t())
# print(dataset[0].x)
# print(dataset[0].edge_attr)
# print(dataset[0].

