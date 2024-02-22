Introduction
------------
1. Here contains all the cif files for three databases.
2. Data collection process for MP, OQMD, CMR and selection process for MP and OQMD. 

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