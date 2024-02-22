import numpy as np 
import os
import pandas as pd
# use Material Project API to get properties
from mp_api.client import MPRester
def get_mp_data():
    key_list=np.load("mp_id.npy").tolist()
    mp=MPRester()
    with MPRester("Cts9tQDKsC5G1PMb9uqBFCSAs0OTJeup") as mpr:
        docs = mpr.summary.search(material_ids=key_list)
    
    #get new id list
    new_id_list=[]
    for i in range(len(docs)):
        new_id_list.append(docs[i].material_id)
    
    def to_cif(docs):
        total_structure=[]
        for i in docs:
            total_structure.append(i.structure)
        total_cif=[]
        for j in total_structure:
            total_cif.append(j.to(fmt="cif",))
        return total_cif    
    cifs=to_cif(docs) # get cif files
    
    # get current path
    current_path=os.getcwd()
    # make dir
    if not os.path.exists("MP"):
        os.makedirs("MP")
    # save cif files
    for j,value in enumerate(cifs) :
        path=current_path+'/MP/'+str(new_id_list[j])+'.cif'
        new_cif=open(path,'w')
        for ii in value:
            new_cif.write(ii)    
        new_cif.close()
    
    data_list=[]
    for i in range(len(docs)):
        data_list.append({
            'provider':'mp',
            'id':docs[i].material_id+'.cif',
            'Full_Formula':docs[i].structure.formula,
            'formula':docs[i].formula_pretty,
            'space_group':docs[i].structure.get_space_group_info()[0],
            'a_lattice_param':docs[i].structure.lattice.a,
            'b_lattice_param':docs[i].structure.lattice.b,
            'c_lattice_param':docs[i].structure.lattice.c,
            'aplha_degree':docs[i].structure.lattice.alpha,
            'beta_degree':docs[i].structure.lattice.beta,
            'gamma_degree':docs[i].structure.lattice.gamma,
            'pbc':docs[i].structure.lattice.pbc,
            'sits':docs[i].structure.sites,
            'volume':docs[i].structure.lattice.volume,
            'species_at_sites':docs[i].structure.species,
            'is_stable':docs[i].is_stable,
            'is_magnetic':docs[i].is_magnetic,    
            'formation_energy':docs[i].formation_energy_per_atom,
            'energy_above_hull':docs[i].energy_above_hull,
            'band_gap':docs[i].band_gap,

        })

    data_df=pd.DataFrame(data_list)
    # data_df.to_csv('mp_perovskite_data_total.csv')
    data_df.to_csv(os.path.join( 'csv_files', "mp_perovskite_data_total.csv"))
    
if __name__ == '__main__':
    get_mp_data()