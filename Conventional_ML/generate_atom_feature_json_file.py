import mendeleev
import numpy as np
import json

# this code show to generate atom_init.json file in Dr.Xie paper "https://doi.org/10.1103/PhysRevLett.120.145301".
# The difference is that instead of using one hot encoding to represent different properties value, here we use its absoublue value to represent its values.

def get_all_feature_list_out(atomic_number=100):
    group_id=[]
    period=[]
    electronegativity=[]
    covalent_radius=[]
    nvalence=[]
    ionenergies=[]
    electron_affinity=[]
    block=[]
    atomic_volume=[]
    
    for i in range(1,atomic_number+1):
        element = mendeleev.element(i)
        group_id.append(element.group_id)
        period.append(element.period)
        electronegativity.append(element.electronegativity())
        covalent_radius.append(element.covalent_radius)
        nvalence.append(element.nvalence())
        ionenergies.append(element.ionenergies[1])
        electron_affinity.append(element.electron_affinity)
        block.append(element.block)
        atomic_volume.append(element.atomic_volume)
    
    replace_none_to_0(group_id)
    replace_none_to_0(period)
    replace_none_to_0(electronegativity)
    replace_none_to_0(covalent_radius)
    replace_none_to_0(nvalence)
    replace_none_to_0(ionenergies)
    replace_none_to_0(electron_affinity)
    replacement_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    block = [replacement_dict[element] for element in block]
    replace_none_to_0(atomic_volume)
    return group_id,period,electronegativity,covalent_radius,nvalence,ionenergies,electron_affinity,block,atomic_volume

def replace_none_to_0(l: list):
    """
    there are are some None value in results such as electronegativity, replace the None valye with 0.
    """
    for i, v in enumerate(l):
        if v is None:
            l[i] = 0
            
def numpy_file_to_json(file):
    """
    input should be a 100 x n np.array
    """
    info={}
    for i in range(1,101):
        info[str(i)]=file[i-1].tolist()
    out_file = open("atom_feature.json", "w")
    json.dump(info, out_file)
    out_file.close()
    
def get_atom_json_out(feature_name:list):
    group_id,period,electronegativity,covalent_radius,nvalence,ionenergies,electron_affinity,block,atomic_volume=get_all_feature_list_out()
    
   
    feature={}
    feature["group_id"]=np.array(group_id).reshape(100,1)
    feature["period"]=np.array(period).reshape(100,1)
    feature["electro_negativity"]=np.array(electronegativity).reshape(100,1)
    feature["covalent_radius"]=np.array(covalent_radius).reshape(100,1)
    feature["valence_electrons"]=np.array(nvalence).reshape(100,1)
    feature["first_ionization_energy"]=np.array(ionenergies).reshape(100,1)
    feature["electron_affinity"]=np.array(electron_affinity).reshape(100,1)
    feature["block"]=np.array(block).reshape(100,1)
    feature["atomic_volume"]=np.array(atomic_volume).reshape(100,1)
    
    start=np.zeros((100,1))
    for i in range(len(feature_name)):
        start=np.concatenate((start,feature[feature_name[i]]),axis=1)
    total_feature=start[:,1:]
    total_feature=total_feature.astype(float)
    numpy_file_to_json(total_feature)
    
    return total_feature
    
def main():
    get_atom_json_out(["group_id","period","electro_negativity","covalent_radius","valence_electrons",
                     "first_ionization_energy","electron_affinity","block","atomic_volume",])

if __name__ == "__main__":
    main()
    