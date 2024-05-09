1. `generate_atom_feature_json_file.py` is use to generate `atom_feature.json` file, which matches section 3.1 structure encoding in the paper. The `atom_feature.json` file contains all the atomic features for atom number range from 1-100. You can also use this python file to generate custom feature design. 

2. `feature_generation.py` take information from `atom_feature.json` and cif file, it then generate features for MP, OQMD and CMR datasets, and save features in '.npy' format.

3. `model_train_mp_fm.py` is an example file used to train model on MP dataset to predict formation energy.

4. Check `CML_main.ipython` to see how to use each function.
