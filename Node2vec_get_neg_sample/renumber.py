import datetime
import random
import time
import json
import pandas as pd
import dgl
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix


def f1(x, dict):
     return dict[x]


def f2(x, dict):
    return dict[str(int(x))]


def getFileColumns(fileName):
    data = pd.read_csv(fileName)
    columns = data.columns
    print(columns)
    column_one = data[columns[0]].to_list()
    column_two = data[columns[1]].to_list()

    return column_one, column_two


edgesTypeDic = {0: [('gene', 'relate1', 'disease')],
                1: [('disease', 'relate2', 'gene')],
                2: [('gene', 'relate3', 'gene')]}




df_tmp_all_node_map = pd.read_csv('../data/all_node_map.txt',header=None,sep='\t')
df_tmp_all_node_map[1] = df_tmp_all_node_map[1].astype(str)
dg_dict_return = dict(zip(df_tmp_all_node_map[1],df_tmp_all_node_map[0]))




_ppi = pd.read_csv('../data/gene_gene.csv')

with open('./Input/emb.txt', 'r') as file:
    lines = file.readlines()

del lines[0]

with open('./Input/emb.txt', 'w') as file:
    file.writelines(lines)

_feature = pd.read_csv('./Input/emb.txt',sep=' ',header=None)

_feature[0] = _feature.apply(lambda x: f2(x[0], dg_dict_return), axis=1)

_file = pd.read_csv('./Input/disease_gene_edges_random_shuffled.csv')

_train = _file

train_g = list(set(_train['gene'].tolist()))

all_gene = train_g
node_dict_g = {}

for n, node in enumerate(all_gene):
    node_dict_g[node] = n

g1 = _ppi['gene1'].tolist()
g2 = _ppi['gene2'].tolist()

g_ = list(set(g1 + g2) - set(train_g))


new_nodes = g_
start_idx = len(node_dict_g)

for m, node in enumerate(new_nodes):
    node_dict_g[node] = m + start_idx


with open('./Output/_dict_gene.txt', 'w') as f:
    json.dump(node_dict_g, f)


train_d = list(set(_train['disease'].tolist()))


gene_list = list(node_dict_g.keys())

result_d = _feature[_feature.iloc[:, 0].isin(train_d)]
result_g = _feature[_feature.iloc[:, 0].isin(gene_list)]



result_d.to_csv('./Output/dis_feature.txt', index=False)
result_g.to_csv('./Output/gene_feature.txt', index=False)

all_dis = result_d[0].tolist()
node_dict_d = {}

for x, node in enumerate(all_dis):
    node_dict_d[node] = x


result_d = result_d.copy()
result_g = result_g.copy()
result_d[0] = result_d.apply(lambda x: f1(x[0], node_dict_d), axis=1)
result_g[0] = result_g.apply(lambda x: f1(x[0], node_dict_g), axis=1)
df_sorted = result_g.sort_values(by=0)


with open('./Output/_dict_d.txt', 'w') as f:
    json.dump(node_dict_d, f)

df_sorted.to_csv('./Output/gene_feature_num.csv', index=False)
result_d.to_csv('./Output/dis_feature_num.csv', index=False)

_ppi = _ppi.copy()
_ppi['gene1'] = _ppi.apply(lambda x: f1(x['gene1'], node_dict_g), axis=1)
_ppi['gene2'] = _ppi.apply(lambda x: f1(x['gene2'], node_dict_g), axis=1)

no_invalid = _file.copy()
no_invalid['disease'] = no_invalid.apply(lambda x: f1(x['disease'], node_dict_d), axis=1)
no_invalid['gene'] = no_invalid.apply(lambda x: f1(x['gene'], node_dict_g), axis=1)

no_invalid.to_csv('./Output/dis_gene_edges_num.csv', index=False)
_ppi.to_csv('./Output/gene_gene_edges_num.csv', index=False)


graph_data = {}

column_one, column_two = getFileColumns('./Output/dis_gene_edges_num.csv')
graph_data.setdefault(edgesTypeDic[0][0],(column_two, column_one))
graph_data.setdefault(edgesTypeDic[1][0],(column_one, column_two))

column_one, column_two = getFileColumns('./Output/gene_gene_edges_num.csv')
graph_data.setdefault(edgesTypeDic[2][0],(column_one+column_two, column_one+column_two))




hg = dgl.heterograph(graph_data)
print(hg)

etype = ('gene', 'relate1', 'disease')
etype2 = ('disease', 'relate2', 'gene')

u, v = hg.edges(etype=etype)

adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
matrix = adj.toarray()

rows, cols = np.where(matrix == 0)

points = list(zip(rows, cols))

data_neg = pd.DataFrame(points)

data_neg.columns = ['gene', 'disease']

data_neg.to_csv('./Output/neg_sample_num.csv', index=False)

f = open('./Output/_dict_gene.txt', 'r')
_dict_gene = eval(f.read())
f.close()

f = open('./Output/_dict_d.txt', 'r')
_dict_d = eval(f.read())
f.close()

data = pd.read_csv('./Input/gene_gene_L3.csv')

data['gene1'] = data['gene1'].map(_dict_gene)
data['gene2'] = data['gene2'].map(_dict_gene)

data.to_csv('./Output/L3_edges_num.csv', index=False)

df1 = pd.read_csv('./Output/gene_gene_edges_num.csv')
df2 = pd.read_csv('./Output/L3_edges_num.csv')
result_L3 = pd.concat([df1, df2], axis=0)
result_L3.to_csv('./Output/gene_gene_all_edges_num.csv', index=False)


data_in = pd.read_csv('../data/disease_gene_edge.csv')
data_in['disease'] = data_in['disease'].map(_dict_d)
data_in['gene'] = data_in['gene'].map(_dict_gene)
data_in.to_csv('./Output/dis_gene_edges_new_num.csv', index=False)
