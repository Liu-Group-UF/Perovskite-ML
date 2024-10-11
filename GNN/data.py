import numpy as np
import pandas as pd 
import os
import json
from pymatgen.core.structure import Structure

import torch
from torch_geometric.data import Data

class GaussianDistance:
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class AtomInitializer:
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class Get_feature_and_Load_Data:
    def __init__(self, csv_name:str='mp_fm.csv', max_num_nbr=12, radius=8, dmin=0, step=0.2):
        path=os.getcwd()
        self.root_dir = os.path.dirname(path.strip()) 
        self.dataset = csv_name.split('_')[0].upper() # get name for dataset
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.id_and_target = pd.read_csv(self.root_dir+'/Data_curation/csv_files/'+csv_name).values
        self.atom_fea = AtomCustomJSONInitializer('atom_init.json')
        self.edge_fea = GaussianDistance(dmin=dmin, dmax=radius, step=step)
    
    def get_atom_and_bond_fea_and_load(self):
        id_cif = self.id_and_target[:,0].tolist()
        target = self.id_and_target[:,1]

        data_list = []
        ## get atom, bond features and edge index to show the connectivity between atoms
        for index, i in enumerate(id_cif):
            crystal = Structure.from_file(os.path.join(self.root_dir,'Data_curation/CIF_files',self.dataset,i))
            atom_fea = np.vstack([self.atom_fea.get_atom_fea(crystal[i].specie.number)
                                  for i in range(len(crystal))])

            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

            nbr_fea = self.edge_fea.expand(nbr_fea)

            atom_fea = torch.Tensor(atom_fea)
            edge_fea = torch.Tensor(nbr_fea).view(-1,41)
            edge_index = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx))
            y = torch.Tensor([float(target[index])]) 
            id = id_cif[index]
         
        ## load data
            graph_crystal = Data(x=atom_fea,edge_index=edge_index,edge_attr=edge_fea,y=y, id=id)
            data_list.append(graph_crystal)
        return data_list
            
    def format_adj_matrix(self,adj_matrix):
        size = len(adj_matrix)
        src_list = list(range(size))
        all_src_nodes = torch.tensor([[x]*adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes = adj_matrix.view(-1).unsqueeze(0)
        return torch.cat((all_src_nodes,all_dst_nodes),dim=0)




