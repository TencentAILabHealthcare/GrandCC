# GrandCC: an Interpretable Graph Attention Networks-based Framework for Cancer Subtype Classification

In this project, we develop a GrandCC model based on a graph attention network to perform molecular subtyping.

The model was validated on two different cancers, including the CMS subtyping of Colorectal  cancer (CRC) 
and the PAM50 classification of breast cancer.

The model was deployed as an easy-to-use online tool, please try the website https://gene.ai.tencent.com/MolecularSubtyping.

## Requirements

```
Python 3.6
PyTorch >= 1.5.0
torch_geometric 1.6.2
numpy
pandas
scipy
sklearn
opencv
random
statsmodels
```


## Task 1. CRCSC subtyping：
The GrandCC model is trained on the TCGA dataset and tested on other 11 datasets. 


### Usage：

+ Validate the model with the 11 test dataset used in our paper, including gse13067, gse13294, gse14333, gse17536, 
  gse20916, gse2109, gse35896, gse37892, gse39582, kfsyscc, petacc3. You can obtain the same results as our paper 
  with the following options:
  
    ```
    cd CRC_GrandCC_opensource
    python3 test_model.py --exp GNN_sim_graph （The pretrained model is saved at: save_models/GNN_sim_graph.pt）
    ```

+ If you want to evaluate our model with your own data, please put your data file in the 'CRC_GrandCC_opensource/
external_dataset' folder (must be in the "txt" format, please follow the details in the below "Data Preparation"). 
Run the following:
    ```
    cd CRC_GrandCC_opensource
    python3 external_test.py --exp GNN_sim_graph --external_file name of your file
    ```
  
  ** The operations in our codes are as follows:
  + The data uploaded by the user must be in "txt" file, and the genes must be represented with entrez ids.
  + Our model will check the overlap of the genes in your uploaded data and our training dataset.
  + If the overlap ratio is less than 70%, the model will exit without making predictions.
  + If the overlap ratio is larger than 70%, the model will match the order of the genes and
    impute the missing genes with all 0, and make predictions using the imputed data.
    



### Data Description：

+ The similarity graph is saved at：./similarity_graphs/TCGA_sim_graph.csv in the following format:

	0 |	1 |	2 | 3 |	... | Num_edge - 1 | Num_edge
  --- | --- | --- | --- | --- | --- | --- 
0	0 |	1 |	1 |	1 | ... | 5972 | 5972
1	0 |	1 |	24 | 43 | ... | 5120 | 5972

    The first row denotes the index of the connected edges. The second and third rows indicate the indices of the 
    two nodes connected by the edge. 


+ The dataset is saved at：./CRC_dataset.
   
   The data structure is：

    ```bash
    ├── gse13067.txt                       
    ├── gse13294.txt
    ├── gse14333.txt
    ├── gse17536.txt
    ├── gse20916.txt
    ├── gse2109.txt 
    ├── gse35896.txt
    ├── gse37892.txt
    ├── gse39582.txt
    ├── kfsyscc.txt
    ├── petacc3.txt
    ├── tcga.txt
    ├── labels.txt
    ```

+ The gene expression data of each dataset is contained in a txt file in the following format.
Each row represents the gene expression of one patient, while each column denotes the expression of one gene (Entrez ID):

    sample| 153396 | 1378 | 5655 | ... | 5345 | 6894 
    --- | --- | --- | --- | --- | --- | --- 
    GSM523257 |5.15212166212432 | 2.88664048871183 | 5.3760491573249 | ... | 6.2371587886039 | 6.4355759773902
    GSM523258 |4.94997883745515 | 2.37108757063653 | 5.0852982410145 | ... | 6.5261487210994 | 6.6052447056984
    ... | ... |	... | ... |	... | ... |	... 
    GSM523263 |5.22185205784889 | 2.20199502833056 | 5.3918173202651 | ... | 6.4885185455419 | 6.9666661456165
    
    ** If you want to evaluate our model with your own data, please prepare the data in this form and make sure the 
    genes are represented in the Entrez ID.

+ The labels of all data are contained in the ./CRC_dataset/labels.txt file, with the column 'CMS_network' used 
to train and evaluate the model performance. The file is in the following format:

    sample | cohort | DeepCC | DeepCC_pre | CMS_network
    --- | --- | --- | --- | --- 
    TCGA-A6-6653 | tcga | CMS1 | CMS1 | CMS1
    TCGA-A6-A56B | tcga | CMS4 | CMS4 | CMS4
    ... | ... | ... | ... | ...
    TCGA-AD-6963 | tcga | CMS2 | CMS2 | CMS2


### Model Prediction：

The predictions of our model are saved in the "./results" folder, in the "csv" form as follows:

sample_id |	CMS1_prob |	CMS2_prob |	CMS3_prob |	CMS4_prob |	GNN_pred | CMS network
--- | --- | --- | --- | --- | --- | --- 
0 |	0.11788527 | 0.6282107 | 0.120226026 | 0.13367797 | 1 | 1
1 | 0.09888291 | 0.66571975 | 0.13151056 | 0.10388676 | 1 | 1
... | ... | ... | ... |	... | ... | ...
N-1 | 0.38062993 | 0.10873932 | 0.07082967 | 0.4398011 | 3 | 0
N |	0.13789913 | 0.33324537 | 0.35867894 | 0.17017654 | 2 | 2

+ Each row represents the prediction of a patient.
+ The 2-5 columns are the predicted probabilities for the four CMS subtypes.
+ The column "GNN_pred" is the class predicted by our model.
+ The column "CMS network" is the ground-truth class.
+ For the prediction of your own data, the result file will not include the last column "CMS network" because 
we don't require the ground-truth labels from you.

### Evaluation：
Please refer to model_evaluation.Rmd file to evaluate the performance.

## Task 2: Breast cancer PAM50 classification：
The GrandCC model is trained on the TCGA dataset and tested on other 5 datasets. 


### Usage：
+ Validate the model with the 5 test dataset used in our paper, including NKI, TRANSBIG, UNT, UPP, VDX. 
You can obtain the same results as our paper with the following options:
    ```
    cd BRCA_GrandCC_opensource
    python3 test_model.py --which_graph similarity --exp GNN_sim_graph --num_nodes 3000
    ```

+ If you want to evaluate our model with your own data, please put your data file in the 'BRCA_GrandCC_opensource/
external_dataset' folder (must be in the "txt" format, please follow the details in the "Data Preparation"). 
Run the following:
    ```
    cd BRCA_GrandCC_opensource
    python3 external_test.py --which_graph similarity --exp GNN_sim_graph --num_nodes 3000 --external_file name of your file
    ```
  
  ** The operations in our codes are as follows:
  + The data uploaded by the user must be in "txt" file, and the genes must be represented with entrez ids.
  + Our model will check the overlap of the genes in your uploaded data and our training dataset.
  + If the overlap ratio is less than 70%, the model will exit without making predictions.
  + If the overlap ratio is larger than 70%, the model will match the order of the genes and
    impute the missing genes with all 0, and make predictions using the imputed data.
    

### Data Description：
+ The dataset is saved at：./BRCA_data_label/intersect_genes.

+ The similarity graph is saved at：./similarity_graphs/TCGA_sim_graph.csv.

+ The file format of the gene expression data and labels are the same as the previously introduced BRCA datasets. 
If you want to evaluate our model with your own data, please make sure to prepare the data in the correct form.



### Model Prediction：

The predictions of our model are saved in the "./results" folder, in the "csv" form as follows:

sample_id |	Basal_prob | LumA_prob | LumB_prob | Her2_prob | Normal_prob |	GNN_pred | label
--- | --- | --- | --- | --- | --- | --- | --- 
NKI_4 | 0.05495929 | 0.02721681 | 0.47734076 | 0.33200058 | 0.10848256 | 2 | 2
NKI_6 | 0.000952676 | 0.79077834 | 0.18619132 | 0.015031756 | 0.007045906 | 1 | 1
... | ... | ... | ... |	... | ... | ... | ...
NKI_8 | 0.8464114 | 0.004770978 | 0.037902106 | 0.0896943 | 0.021221258 | 0 | 0
NKI_11 | 0.060146123 | 0.13140255 | 0.47476345 | 0.30161282 | 0.03207499 | 2 | 3


+ Each row represents the prediction of a patient.
+ The 2-6 columns are the predicted probabilities for the five PAM50 subtypes.
+ The column "GNN_pred" is the class predicted by our model.
+ The column "label" is the ground-truth class.
+ For the prediction of your own data, the result file will not include the last column "label" because 
we don't require the ground-truth labels from you.

### Disclaimer

This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.

### Copyright

This tool is developed in Tencent AI Lab. 

The copyright holder for this project is Tencent AI Lab. 

All rights reserved.
