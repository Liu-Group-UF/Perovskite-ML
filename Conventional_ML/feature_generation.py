import pandas as pd
import numpy as np
import warnings
import os
from glob import glob
import json
from pymatgen.core.structure import Structure
import argparse

# we use some Dr.Xie's code here. https://github.com/txie-93/cgcnn

class GaussianDistance:
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
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
            
def bond_feature(crystal, cif_id):
    max_num_nbr = 12
    radius = 8.0
    dmin = 0
    step = 0.2
    all_nbrs = crystal.get_all_neighbors(8.0, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            warnings.warn('{} not find enough neighbors to build graph. '
                          'If it happens frequently, consider increase '
                          'radius.'.format(cif_id))
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                               [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                           [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    return nbr_fea.reshape(-1)

def get_all_features(database="MP"):
    path=os.getcwd()
    parent_directory = os.path.dirname(path.strip())
    if database=="OQMD":
        id_list=pd.read_csv(parent_directory +"/Data/csv_files/"+ database.lower() + "_fm_modified.csv")["id"]
    else:
        id_list=pd.read_csv(parent_directory +"/Data/csv_files/"+ database.lower() + "_fm.csv")["id"]
    atom_file = 'atom_abs_value.json'
    assert os.path.exists(atom_file), 'atom_abs_value.json does not exist!'
    ari = AtomCustomJSONInitializer(atom_file)
    # get cif file path
    cif_path=os.path.join(parent_directory,"Data/CIF_files",database) 
    # combine bond and atom features
    feature_all=[]
    for i in id_list:
        crystal = Structure.from_file(os.path.join(cif_path,i)) 
        bond=bond_feature(crystal,i)
        atom_fea = np.concatenate([ari.get_atom_fea(crystal[i].specie.number)
                                  for i in range(len(crystal))])  
        total_fea=np.concatenate((atom_fea,bond))
        feature_all.append(total_fea) 
    # stack to get an array
    final_feature=np.vstack([j for j in feature_all])
    np.save(database.lower()+"_features.npy",final_feature)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate features for conventional machine learning models for differenrt databases')
    parser.add_argument('-d','--database', choices=['MP','OQMD','CMR'], metavar="", default="MP",
                        help='database to use, i.e., MP, OQMD, CMR')
    args = parser.parse_args()
    get_all_features(args.database)