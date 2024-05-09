Introduction
------------
1. Here contains all the cif files for three databases.
2. `data_collect_mp.py` and `data_collection.py` stands for data collection process for MP, OQMD, CMR. This matches first and second steps in figure 1 of paper. 
3. `data_selection.py` stands for data selection process for MP and OQMD, it matches step 3 and step 4 in figure 1 of paper. 

Dependencies
------------
env 1:
-  Python3
-  Numpy
-  Pandas
-  Pymatgen==2022.11.7
-  Matminer

env 2:
-  Python3
-  Numpy
-  Pandas
-  Pymatgen

Usage:
-------------
1. Create a virtual enviorment 1, install the packge then run "python data_collection.py" in the termial to collect oqmd and cmr data, it will generate cif files for structure and csv file for properties. 
2. Create a new environemnt 2 to install latest version of pymatgen, then run "python data_collect_mp.py" in the termial to get mp data including cif files and csv file.
3. Run "python data_selection.py" in the termial to get prepared csv files for selected perovskite data and corresponding csv files.