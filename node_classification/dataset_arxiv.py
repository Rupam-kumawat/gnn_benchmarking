



import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS
from ogb.nodeproppred import NodePropPredDataset
# from ogb.nodeproppred import NodePropPredDataset

import zipfile
import numpy as np
import scipy.sparse as sp
from os import path, makedirs
#from google_drive_downloader import GoogleDriveDownloader as gdd
import gdown
import scipy
from torch_geometric.datasets import Planetoid
from data_utils_arxiv import dataset_drive_url, rand_train_test_idx

import os
import zipfile

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

import os
import zipfile
import numpy as np
import torch
from torch_geometric.data import Data
import gdown

from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])
import shutil


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                         but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}

        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset
        Returns NCDataset
    """
    print(dataname)
    if dataname in ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, dataname)
    elif dataname in ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, dataname)
    elif dataname in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        dataset = load_hetero_dataset(data_dir, dataname)
    elif dataname == 'wikics':
        dataset = load_wikics_dataset(data_dir)
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname)
    # Added 'squirrel-filtered' to this condition
    elif dataname in ('chameleon', 'squirrel', 'squirrel-filtered'):
        # Map 'squirrel-filtered' to 'squirrel' as the loader expects this name
        name_for_path = 'squirrel' if dataname == 'squirrel-filtered' else dataname
        dataset = load_wiki_new(data_dir, name_for_path)
    # Added loader for 'wiki-cooc'
    elif dataname == 'wiki-cooc':
        dataset = load_wiki_cooc_dataset(data_dir)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_planetoid_dataset(data_dir, name, no_feat_norm=True):
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labelf

    return dataset


def load_wiki_new(data_dir, name):
    path = f'{data_dir}/geom-gcn/{name}/{name}_filtered.npz'
    data = np.load(path)
    # lst=data.files
    # for item in lst:
    #     print(item)
    node_feat = data['node_features']  # unnormalized
    labels = data['node_labels']
    edges = data['edges']  # (E, 2)
    edge_index = edges.T

    dataset = NCDataset(name)

    edge_index = torch.as_tensor(edge_index)
    node_feat = torch.as_tensor(node_feat)
    labels = torch.as_tensor(labels)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels

    return dataset


def load_wikics_dataset(data_dir):
    wikics_dataset = WikiCS(root=f'{data_dir}/wikics/')
    data = wikics_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset('wikics')
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_hetero_dataset(data_dir, name):
    #transform = T.NormalizeFeatures()
    torch_dataset = HeterophilousGraphDataset(
        name=name.capitalize(), root=data_dir)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    ## dataset splits are implemented in data_utils.py
    '''
    dataset.train_idx = torch.where(data.train_mask[:,0])[0]
    dataset.valid_idx = torch.where(data.val_mask[:,0])[0]
    dataset.test_idx = torch.where(data.test_mask[:,0])[0]
    '''

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Photo', transform=transform)
    elif name == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Computers', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='CS', transform=transform)
    elif name == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='Physics', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    with torch.serialization.safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr]):
        print('data_dir/ogb', f'{data_dir}/ogb')
        shutil.rmtree(f'{data_dir}/ogb')
        ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(
        dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx
    dataset.load_fixed_splits = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


# def load_wiki_cooc_dataset(data_dir):
#     """
#     Loads the wiki-cooc dataset.
#     Downloads files if they are not found.
#     Uses an identity matrix for node features.
#     """
#     dataname = 'wiki-cooc'
#     data_path = f'{data_dir}/{dataname}'
    
#     # Check if directory exists, if not create it
#     if not path.exists(data_path):
#         makedirs(data_path)

#     npz_file_path = f'{data_path}/{dataname}.npz'
#     npy_file_path = f'{data_path}/{dataname}_label.npy'

#     # Download data files if they don't exist
#     if not path.exists(npz_file_path):
#         print(f"Downloading {dataname}.npz...")
#         url_npz = 'https://raw.githubusercontent.com/wzzlcss/Heterophilic_Benchmarks/main/data/wiki-cooc/wiki-cooc.npz'
#         gdown.download(url=url_npz, output=npz_file_path, quiet=False)
        
#     if not path.exists(npy_file_path):
#         print(f"Downloading {dataname}_label.npy...")
#         url_npy = 'https://raw.githubusercontent.com/wzzlcss/Heterophilic_Benchmarks/main/data/wiki-cooc/wiki-cooc_label.npy'
#         gdown.download(url=url_npy, output=npy_file_path, quiet=False)

#     # Load data from files
#     # data = np.load(npz_file_path)
#     # labels = np.load(npy_file_path)
    
#     # NEW CODE
#     data = np.load(npz_file_path, allow_pickle=True)
#     labels = np.load(npy_file_path, allow_pickle=True)
    
#     # Reconstruct sparse adjacency matrix
#     adj = sp.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
#     num_nodes = adj.shape[0]

#     # Create identity matrix for node features as they are not provided
#     node_feat = torch.eye(num_nodes)

#     # Convert adjacency matrix to edge index format
#     adj_coo = adj.tocoo()
#     edge_index = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col))).long()

#     # Convert labels to tensor
#     labels = torch.from_numpy(labels).long()
    
#     # Create and populate NCDataset object
#     dataset = NCDataset(dataname)
#     dataset.graph = {
#         'edge_index': edge_index,
#         'node_feat': node_feat,
#         'edge_feat': None,
#         'num_nodes': num_nodes
#     }
#     dataset.label = labels
    
#     return dataset

# Add this import at the top of your file with the other imports



def load_wiki_cooc_dataset(data_dir: str):
    dataname = 'wiki_cooc'
    zip_filename = 'data.zip'
    zip_file_path = os.path.join(data_dir, zip_filename)
    npz_file = os.path.join(data_dir, f"{dataname}.npz")

    # 1. Ensure dataset exists
    if not os.path.exists(npz_file):
        os.makedirs(data_dir, exist_ok=True)
        print(f"'{dataname}.npz' not found. Downloading '{zip_filename}'...")
        zip_url = (
            "https://github.com/wzzlcss/Heterophilic_Benchmarks/raw/"
            "030f1f657c4d61cdcfb400595d96f9d250f956de/Opengsl/data.zip"
        )
        gdown.download(url=zip_url, output=zip_file_path, quiet=False)

        print(f"Extracting {zip_file_path}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"Failed to find '{npz_file}' after extraction.")

    # 2. Load the .npz
    data = np.load(npz_file, allow_pickle=True)
    print("Loaded keys:", data.files)

    # 3. Extract arrays
    x = torch.tensor(data['node_features'], dtype=torch.float32)
    y = torch.tensor(data['node_labels'], dtype=torch.long)

    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()

    # train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
    # val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
    # test_mask = torch.tensor(data['test_masks'], dtype=torch.bool)

    # 4. Build PyG Data object
    # dataset = Data(
    #     x=x,
    #     edge_index=edge_index,
    #     label=y,
    #     train_mask=train_mask,
    #     val_mask=val_mask,
    #     test_mask=test_mask,
    # )
    # dataset.num_nodes = x.size(0)
    
    dataset= NCDataset(dataname)
    
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': x,
                     'num_nodes': x.size(0)}

    # label = data['node_labels'].flatten()
    dataset.label = torch.tensor(y, dtype=torch.long)

    return dataset





def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
        gdown.download(id=drive_id, output="data/pokec/pokec.mat")
        #import sys; sys.exit()
        #gdd.download_file_from_google_drive(
        #    file_id= drive_id, \
        #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    return dataset