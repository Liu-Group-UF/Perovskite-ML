from pymatgen.ext.optimade import OptimadeRester
from matminer.datasets import load_dataset
import pandas as pd
import numpy as np
import os

# get data from oqmd
def get_data_from_oqmd():
    opt_oqmd=OptimadeRester(['oqmd'],timeout=1800)
    # if need to get all anonymous with A3BC, delete nsites=5 in the search condition
    result_band=opt_oqmd.get_snls(chemical_formula_anonymous="A3BC",nsites=5,additional_response_fields=['_oqmd_band_gap',
                                                                                        '_oqmd_stability',
                                                                                        '_oqmd_delta_e',
                                                                                       '_oqmd_spacegroup'])
    key_list=list(result_band['oqmd'].keys())
    total_data=[]
    for index,key in enumerate(key_list):
        total_data.append(result_band['oqmd'][key].structure)

    total_cif=[]
    for i in total_data:
        total_cif.append(i.to(fmt="cif"))
    
    # get current path
    current_path=os.getcwd()
    # make dir
    if not os.path.exists("OQMD"):
        os.makedirs("OQMD")
    # save cif files
    for j,value in enumerate(total_cif) :
        path=current_path+'/OQMD/'+str(key_list[j])+'.cif'
        new_cif=open(path,'w')
        for ii in value:
            new_cif.write(ii)    
        new_cif.close()
    
    # save inforamtion to csv file
    data_list=[]
    for i in result_band['oqmd'].keys():
        data_list.append({
            'provider':'oqmd',
            'id':str(i)+".cif",
            'Full_Formula':result_band['oqmd'][i].structure.formula,
            'formula':result_band['oqmd'][i].structure.composition.reduced_formula,
            'space_group':result_band['oqmd'][i].structure.get_space_group_info()[0],
            'a_lattice_param':result_band['oqmd'][i].structure.lattice.a,
            'b_lattice_param':result_band['oqmd'][i].structure.lattice.b,
            'c_lattice_param':result_band['oqmd'][i].structure.lattice.c,
            'aplha_degree':result_band['oqmd'][i].structure.lattice.alpha,
            'beta_degree':result_band['oqmd'][i].structure.lattice.beta,
            'gamma_degree':result_band['oqmd'][i].structure.lattice.gamma,
            'pbc':result_band['oqmd'][i].structure.pbc,
            'sits':result_band['oqmd'][i].structure.sites,
            'volume':result_band['oqmd'][i].structure.lattice.volume,
            'species_at_sites':result_band['oqmd'][i].data['_optimade']['species_at_sites'],
            'formation_energy':result_band['oqmd'][i].data['_optimade']['_oqmd_delta_e'],
            'stability':result_band['oqmd'][i].data['_optimade']['_oqmd_stability'],
            'band_gap':result_band['oqmd'][i].data['_optimade']['_oqmd_band_gap'],
        })

    data_df=pd.DataFrame(data_list)
    # make dir to save csv file
    if not os.path.exists("csv_files"):
        os.makedirs("csv_files")

    data_df.to_csv(os.path.join( 'csv_files', 'oqmd_perovskite_data_total.csv'))

# get data from mp
def get_data_from_mp():
    # get name list 
    opt1=OptimadeRester(['mp'],timeout=1800)
    # if need to get all anonymous with A3BC, delete nsites=5 in the search condition
    result1=opt1.get_structures(chemical_formula_anonymous="A3BC",nsites=5)
    key_list=list(result1['mp'].keys())
    np.save("mp_id.npy",key_list)

def get_data_from_cmr():
    # get the csv file
    df=load_dataset("castelli_perovskites")
    df_new=df.drop(columns="structure") # drop structure since what it contains can be seen in cif files
    
    # create id list
    id_list=[]
    for i in range(len(df)):
        name="perov_"+str(i)+(".cif")
        id_list.append(name)
    df_new.insert(0,"id",id_list)
    df_new.to_csv(os.path.join( 'csv_files', "cmr_perovskite_data_total.csv"))
    # df_new.to_csv("cmr_perovskite_data_total.csv")
    
    # get cif files
    total_cif=[]
    for i in df['structure']:
        total_cif.append(i.to(fmt="cif"))    
    current_path=os.getcwd()
    for j,value in enumerate(total_cif) :
        path_new=current_path+'/CMR/'+df_new['id'][j]
        new_cif=open(path_new,'w')
        for ii in value:
            new_cif.write(ii)    
        new_cif.close()
                            
def main():
    get_data_from_oqmd()
    get_data_from_mp()
    get_data_from_cmr()

    
if __name__ == '__main__':
    main()    
