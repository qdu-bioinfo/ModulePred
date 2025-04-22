# ModulePred

## Contents

- <u>[Introduction](#Introduction)</u>

- <u>[Package](#Package)</u>

- <u>[Installation](#Installation)</u>

- <u>[Execution](#Execution)</u>

- <u>[Contact](#Contact)</u>

## Introduction

In this study, we introduce ModulePred, a sophisticated deep learning framework designed to enhance the prediction of disease-gene associations. ModulePred innovatively amalgamates existing disease-gene associations to construct intricate heterogeneous module network complexes involving proteins. This approach not only enriches protein interactions but also pioneers the development of a novel heterogeneous module network graph embedding technique. These advancements collectively contribute to a significant improvement in the performance of predicting correlations between diseases and genes

## Package

torch==1.13.1

dgl==1.1.2

gensim==4.2.0

tqdm==4.64.1

node2vec==0.4.3

scikit-learn==1.0.2

## Installation

```
cd ModulePred-main
sh init.sh
```

## Execution

If you want to predict pathogenic genes for specific diseases of interest (by inputting DISGENET disease IDs into data/target_diseases.csv; https://disgenet.com/), you can directly run disease_gene_prediction.py in the Disease_gene_prediction folder. The prediction results will be saved in the GNN/Output folder.



You can also follow the full workflow as described below:

(1) In the data folder:

- disease_gene_edge.csv: Associations between genes and diseases.

- gene_gene.csv: Protein-protein interactions (proteins have been mapped to genes).

- all_node_map.txt: A dictionary mapping genes and diseases to unique IDs.

- all_human_complexes.csv: Protein complexes.

- target_diseases.csv: specific diseases of interest

(2)  L3 (First,run map_gene_edges.py,Second run the script generate_new_edges.py to generate L3, finally run get_all_edges.py):

In the Output folder:

- gene_gene_L3.csv: Protein-protein interactions enhanced through graph data augmentation.

- all_node_edge_num.txt: Contains all edges (gene_gene.csv + gene_gene_L3.csv + disease_gene_edge.csv), with nodes mapped to IDs based on all_node_map.txt.

- gene_mapping.txt : is the mapping of gene nodes in the original gene edge network, with a total of 15964 genes in the network

- gene_num_return_dict.txt : is a reflection of gene_mapping

- gene_gene_num.csv : is the mapping of the edges of the original gene

- pairs_L3.csv : is the edge selected by CN

- path_L3.csv : is the intermediate file of CN_stcore.exe runtime

- path_L3_SA.csv : is the edge selected for RA

- path_L3_AA.csv : is the edge selected for AA

- gene_gene_L3.csv : is the protein interaction enhanced by graph data

- all_node_edge_num.txt : contains all connected edges (gene_gene. csv+gene_gene_L3. csv+disease_gene. edge. csv), and maps nodes with IDs according to all_node_map.txt
  
  

(3) Node2vec_gen_walks (Run e2v_walks.py):

In the Input folder:

- all_node_edge_num.txt: Output from L3/Output folder.

In the Output folder:

- walks.txt: Paths generated through random walks.

(4) Node2vec_learn_vec (Run learn_vecs.py):

In the Input folder:

- walks.txt: Output from Node2vec_gen_walks/Output folder.

In the Output folder:

- emb.txt: Low-dimensional node embeddings.

- emb.model: Trained Node2Vec model file.
  
  

(5) Node2vec_get_neg_sample (Run renumber.py):

In the Input folder:

- disease_gene_edges_random_shuffled.csv: Randomly shuffled version of disease_gene_edge.csv.

- emb.txt: From Node2vec_learn_vec/Output.

- gene_gene_L3.csv: From L3/Output.

In the Output folder:

- _dict_d.txt: Dictionary mapping disease nodes.

- _dict_gene.txt: Dictionary mapping gene nodes.

- dis_feature.txt, dis_feature_num.csv: 128-dimensional features for diseases from Node2Vec.

- gene_feature.txt, gene_feature_num.csv: 128-dimensional features for genes from Node2Vec.

- dis_gene_edges_num.csv: Disease-gene associations with nodes mapped using _dict_d and _dict_gene,The order of edges has been disrupted

- dis_gene_edges_new_num.csv: Disease-gene associations with nodes mapped using _dict_d and _dict_gene,The order of edges is not disrupted

- gene_gene_all_edges_num.csv: All gene-gene edges, mapped using _dict_gene.

- gene_gene_edges_num.csv: Original gene-gene edges, mapped using _dict_gene.

- L3_edges_num.csv: Graph-augmented gene-gene edges, mapped using _dict_gene.

- neg_sample_num.csv: Negative samples for Disease_gene_prediction.
  
  

(6) Node2vec_get_Module_Embedding (First run process_complexes.py, then get_Module_Embeeding.py):

In the Input folder:

- _dict_d.txt, _dict_gene.txt: From Node2vec_get_neg_sample output.

- disease_gene_edges_random_shuffled.csv: Randomly shuffled version of disease_gene_edge.csv.

- gene_gene_L3.csv: From L3/Output.

In the Output folder:

- dis_EmbeddingsDic: 128-dimensional disease features.

- dis_feature_256, dis_feature_256_num: 256-dimensional disease features.

- gene_EmbeddingsDic: 128-dimensional gene features.

- gene_feature_256, gene_feature_256_num: 256-dimensional gene features.

(7) Disease_gene_prediction (Run disease_gene_prediction.py)

In the Input folder:

- dis_gene_edges_new_num.csv, gene_gene_all_edges_num.csv, gene_feature_num.csv, dis_feature_num.csv, neg_sample_num.csv, _dict_d.txt, _dict_gene.txt: From Node2vec_get_neg_sample output.

- gene_feature_256_num.csv, dis_feature_256_num.csv: From Node2vec_get_Module_Embedding output.

In the Output folder:

Candidate disease-associated genes for each disorder listed in 'data/target_diseases.csv'.



## Contact

All problems please contact ModulePred development team: **Xiaoquan Su**    Email: [suxq@qdu.edu.cn](mailto:suxq@qdu.edu.cn)
