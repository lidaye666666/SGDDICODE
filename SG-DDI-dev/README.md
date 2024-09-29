# SG-DDI



## Requirements  

numpy==1.18.1 \
tqdm==4.42.1 \
pandas==1.0.1 \
rdkit==2009.Q1-1 \
scikit_learn==1.0.2 \
torch==1.11.0 \
torch_geometric==2.0.4 \
torch_scatter==2.0.9

## Step-by-step running:  
### 1. DrugBank
- First, cd SG-DDI/drugbank, and run data_preprocessing.py using  
  `python data_preprocessing.py -d drugbank -o all`  
  Running data_preprocessing.py convert the raw data into graph format. \
   Create a directory using \
  `mkdir save`  
  
- Second, run train.py using 
  `python train.py --fold 0 --save_model` 

  to train SG-DDI. The training record can be found in save/ folder.







### 2. TWOSIDES
- First, cd SG-DDI/drugbank, and run data_preprocessing.py using  
  `python data_preprocessing.py -d twosides -o all`   
  Running data_preprocessing.py convert the raw data into graph format.
  Create a directory using \
  `mkdir save`
- Second, run train.py using 
  `python train.py --fold 0 --save_model` 

  to train SG-DDI. The training record can be found in save/ folder.
