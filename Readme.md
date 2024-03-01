# ModulePred

## Contents

- <u>[Introduction](#Introduction)</u>

- <u>[Package](#Package)</u>

- <u>[Installation](#Installation)</u>

- <u>[Data](#Data)</u>

- <u>[Training](#Training)</u>

- <u>[Contact](#Contact)</u>

## Introduction

In this study, a deep learning framework called ModulePred was introduced to predict disease gene associations. Module Pred uses the L3 link prediction algorithm to graphically enhance protein interaction networks. It integrates disease gene associations, constructs heterogeneous module network complexes with proteins, and enhances protein interactions, and develops a new heterogeneous module network graph embedding. Improved the performance of predicting the correlation between diseases and genes

## Package

torch==1.13.1

dgl==1.1.2

gensim==4.2.0

tqdm==4.64.1

node2vec==0.4.3

scikit-learn==1.0.2

## Installation

```
cd process
sh init.sh
```

## Data

In this demo, we constructed a virtual dataset to run the program example normally. The original dataset is stored in the /data/raw_data directory.
a.    gene_gene.csv (The correlation between genes in a network diagram)

| gene1 | gene2 |
| ----- | ----- |
| g0    | g1    |
| g0    | g4    |
| g3    | g6    |

b.    gene_num_return_dict.txt (Gene Number Dictionary)

| gene | number |
| ---- | ------ |
| g0   | 0      |
| g1   | 1      |
| g2   | 2      |

c.    used_human_complexes.csv (Gene and disease motif information)

| ComplexID | subunits(Gene name) | Disease |
| --------- | ------------------- | ------- |
| 1         | g0;g1               | d0      |
| 2         | g2;g3               | NA      |
| 3         | g4;g5               | NA      |

d.gene_disease.csv (The connection between disease and genes)

| Disease | Gene |
| ------- | ---- |
| d0      | g0   |
| d0      | g1   |
| d0      | g2   |

Training
-----------------------------

For the original data set in this demo, you can run the program with one click through the configured .sh file to generate a model prediction file.Through graph training, the score of possible connections between disease node d0 and gene nodes g4, g5, g6, g7, g8, and g9 is obtained.

### Network Diagram

<img src="file:///C:/Users/Administrator/%E6%A1%8C%E9%9D%A2/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20240301142822.png" title="" alt="" style="zoom:33%;">

For convenience, you can run the processes above by running the run.sh in folder 'process/'.

```
cd process
chmod a+x run.sh
./run.sh
```

### forecast result

An example of the prediction result file is shown in the chart

| disease | gene | score      |
| ------- | ---- | ---------- |
| d0      | g5   | 0.30815384 |
| d0      | g6   | 0.28945717 |
| d0      | g7   | 0.2887915  |
| d0      | g8   | 0.3041018  |
| d0      | g9   | 0.29970533 |

## Contact

All problems please contact WTHMDA development team: **Xiaoquan Su**    Email: [suxq@qdu.edu.cn](mailto:suxq@qdu.edu.cn)
