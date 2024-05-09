from pymatgen.analysis.local_env import VoronoiNN,site_is_of_motif_type
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
import pandas as pd
import os
from glob import glob

def get_rid_of_X_3(file):
    # construct a list: ['In', 'Sn', 'Br', 'Br', 'Br']
    info_list=[]
    for i in file.sites:
        element=str(i).split("] ")[-1]
        info_list.append(element)
    
    # get count for atom: {'In': 1, 'Sn': 1, 'Br': 3}
    element_counts = {}
    # Count the occurrences of each element in the original list
    for element in info_list:
        element_counts[element] = element_counts.get(element, 0) + 1
       
    # output index list
    index_list=[0,1,2,3,4]
    output_index=[]
    for i, j in zip (index_list,info_list):
        if element_counts[j]==3:
            continue
        output_index.append(i)
    return output_index # output should be a list only contains tow index e.g. [0,1]

def get_octahedral(file_list, file_path="./MP"):    
    cif_list=[]
    for i in sorted(file_list):
        structure=Structure.from_file(os.path.join(file_path,i))
        cif_list.append(structure)
       
    new_cif_name=[]
    for i,j in zip(sorted(file_list), cif_list):       
        # get the selected atom index list
        atom_index = get_rid_of_X_3(j)  # center atom list e.g. [0,1] since we don't sure which atom is the center
        
        for index in atom_index:
            criteria=site_is_of_motif_type(j,index,delta=0.1,approach="voronoi")
            if criteria=="octahedral":
                new_cif_name.append(i)
        
    return new_cif_name

# select by using structure match
def select_by_structure_match(file_list,file_path="./MP/"):
    # get prototype
    prototype_cif=glob("./prototype/*.cif")
    # match
    matcher=StructureMatcher()
    mp_new=[]
    for i in file_list:
        new_path=file_path+i
        structure_2=Structure.from_file(new_path)
        for j in prototype_cif:
            structure_1=Structure.from_file(j)
            if matcher.fit_anonymous(structure_1,structure_2)==True:
                mp_new.append(i)
    return np.unique(np.array(mp_new)).tolist()

def get_new_csv(mp_new,mp_total,prop):
    prop_value=[]
    mp_new_2=[]
    for i,j in enumerate(mp_total["id"]):
        if j in mp_new:
            prop_value.append(mp_total[prop][i])
            mp_new_2.append(j)
    info={}
    info["id"]=mp_new_2
    info[prop]=prop_value
    df=pd.DataFrame(info)
    return df

def main(): # select crytsal data to get perovskite data with its property and save it as csv file
    mp_total=pd.read_csv("csv_files/mp_perovskite_data_total.csv")
    oqmd_total=pd.read_csv("csv_files/oqmd_perovskite_data_total.csv")
    cmr_total=pd.read_csv("csv_files/cmr_perovskite_data_total.csv")
    
    mp_id_list=mp_total["id"].values.tolist()
    oqmd_id_list=oqmd_total["id"].values.tolist()
    
    mp_selected=select_by_structure_match(get_octahedral(mp_id_list)) # select by octahedral first and then strucutre match
    oqmd_selected=select_by_structure_match(get_octahedral(oqmd_id_list,file_path="./OQMD"),file_path="./OQMD/")
    # save it to csv file
    # mp
    get_new_csv(mp_selected,mp_total,"band_gap").to_csv("csv_files/mp_bg.csv",index=False)
    get_new_csv(mp_selected,mp_total,"formation_energy").to_csv("csv_files/mp_fm.csv",index=False)
    # oqmd
    get_new_csv(oqmd_selected,oqmd_total,"band_gap").to_csv("csv_files/oqmd_bg.csv",index=False)
    get_new_csv(oqmd_selected,oqmd_total,"formation_energy").to_csv("csv_files/oqmd_fm.csv",index=False) 
    # cmr
    pd.concat([cmr_total["id"],cmr_total["e_form"]],axis=1).to_csv("csv_files/cmr_fm.csv",index=False)
    pd.concat([cmr_total["id"],cmr_total["gap gllbsc"]],axis=1).to_csv("csv_files/cmr_bg.csv",index=False)
    
if __name__ == '__main__':
    main()    
    
    
    
    
    
    