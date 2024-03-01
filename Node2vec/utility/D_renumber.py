import datetime
import random
import time
import json

import pandas as pd
from tqdm import tqdm

import dgl
import dgl.nn as dglnn
import dgl.function as fn

import torch.nn as nn
import torch.nn.functional as F
import torch

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

f = open('./data/Node2vec/dis_gene_dict_return.txt', 'r')
dg_dict_return = eval(f.read())
f.close()



#----------------------------------------------------------------------------------------------------------------------------
# 对于基因疾病，对他们进行重新编号，要将致病基因放在字典前面，同时新字典处理好后，寻找疾病基因的负样本（负连边）

_ppi = pd.read_csv('./data/Node2vec/gene_gene.csv')
# 读取原始txt文件
with open('./data/Node2vec/emb.txt', 'r') as file:
    lines = file.readlines()

# 删除第一行
del lines[0]

# 将修改后的内容写回文件
with open('./data/Node2vec/emb.txt', 'w') as file:
    file.writelines(lines)

_feature = pd.read_csv('./data/Node2vec/emb.txt',sep=' ',header=None)

#   使用之前定义的 f2 函数，将 _feature 中的第一列映射为基因/疾病的原始名称

_feature[0] = _feature.apply(lambda x: f2(x[0], dg_dict_return), axis=1)

#   读取划分的10折文件
_file = pd.read_csv('./data/Node2vec/new_cv.txt')
_train = _file[_file['index'] == 'train']

# 得到所有的致病基因数
train_g = list(set(_train['gene'].tolist()))
print('所有的致病基因数目：', len(train_g))

#   生成基因编号字典：致病基因在前，其余基因在后
#   必须要重新生成字典，矩阵负样本采集，需要用到这个字典
all_gene = train_g
node_dict_g = {}

for n, node in enumerate(all_gene):
    node_dict_g[node] = n

g1 = _ppi['gene1'].tolist()
g2 = _ppi['gene2'].tolist()

g_ = list(set(g1 + g2) - set(train_g))

print('所有的非致病基因数目：', len(g_))

new_nodes = g_
start_idx = len(node_dict_g)

for m, node in enumerate(new_nodes):
    node_dict_g[node] = m + start_idx


#   保存这一折中包含的基因的字典文件:_dict_gene
with open('./data/Node2vec/_dict_gene.txt', 'w') as f:
    json.dump(node_dict_g, f)

print('所有的字典中基因数目：', len(node_dict_g))
print('-------基因字典编码文件处理完毕---------')

#   筛选出基因特征和疾病特征，分别保留
train_d = list(set(_train['disease'].tolist()))
print('所有的疾病数目：', len(train_d))



result_d = _feature[_feature.iloc[:, 0].isin(train_d)]
result_g = _feature[~_feature.iloc[:, 0].isin(train_d)]


result_d.to_csv('./data/Node2vec/dis_feature.txt', index=False)
result_g.to_csv('./data/Node2vec/gene_feature.txt', index=False)

print('-------疾病、基因初始文件特征保存处理完毕---------')

  # 疾病的编码
all_dis = result_d[0].tolist()
node_dict_d = {}

for x, node in enumerate(all_dis):
    node_dict_d[node] = x

result_d = result_d.copy()
result_g = result_g.copy()
result_d[0] = result_d.apply(lambda x: f1(x[0], node_dict_d), axis=1)
result_g[0] = result_g.apply(lambda x: f1(x[0], node_dict_g), axis=1)
df_sorted = result_g.sort_values(by=0)

# 保存疾病字典文件
with open('./data/Node2vec/_dict_d.txt', 'w') as f:
    json.dump(node_dict_d, f)
  # 保存疾病、基因特征文件
df_sorted.to_csv('./data/Node2vec/gene_feature_num.csv', index=False)
result_d.to_csv('./data/Node2vec/dis_feature_num.csv', index=False)

print('-------疾病、基因初始文件映射对应ID后的特征保存处理完毕---------')

# 映射两个文件--疾病-基因、ppi
#   将之前的疾病、基因节点编号映射为新的疾病、基因节点编号
_ppi = _ppi.copy()
_ppi['gene1'] = _ppi.apply(lambda x: f1(x['gene1'], node_dict_g), axis=1)
_ppi['gene2'] = _ppi.apply(lambda x: f1(x['gene2'], node_dict_g), axis=1)

no_invalid = _file[_file['index'] != 'invalid'].copy()
no_invalid['disease'] = no_invalid.apply(lambda x: f1(x['disease'], node_dict_d), axis=1)
no_invalid['gene'] = no_invalid.apply(lambda x: f1(x['gene'], node_dict_g), axis=1)

no_invalid.to_csv('./data/Node2vec/dis_gene_edges_num.csv', index=False)
_ppi.to_csv('./data/Node2vec/gene_gene_edges_num.csv', index=False)

print('-------ppi文件、dis_gene文件映射，保存处理完毕---------')

print(
    '========================================================得到负样本矩阵文件....=========================================================')

graph_data = {}

column_one, column_two = getFileColumns('./data/Node2vec/dis_gene_edges_num.csv')
graph_data.setdefault(edgesTypeDic[0][0],(column_two, column_one))
graph_data.setdefault(edgesTypeDic[1][0],(column_one, column_two))

column_one, column_two = getFileColumns('./data/Node2vec/gene_gene_edges_num.csv')
graph_data.setdefault(edgesTypeDic[2][0],(column_one+column_two, column_one+column_two))



# disease_gene = pd.read_csv('./data/Node2vec/dis_gene_edges_num.csv')

hg = dgl.heterograph(graph_data)
print(hg)
# 给边编号
etype = ('gene', 'relate1', 'disease')
etype2 = ('disease', 'relate2', 'gene')

#=================================================================================================================================================
# 正样本
# u, v = hg.edges(etype=etype)  # 源节点 尾节点
# print(u,v)
# 矩阵
# adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
# matrix = adj.toarray()
# print(matrix)

# 获取全为0的行坐标和列坐标
# rows, cols = np.where(matrix == 0)
# 将行坐标和列坐标组成点对，并存储在一个列表里
# points = list(zip(rows, cols))
# print(points)
# data_neg = pd.DataFrame(points)

# data_neg.columns = ['gene', 'disease']

# data_neg.to_csv('./data/Node2vec/neg_sample_num.csv', index=False)
# print('预处理过程结束！')
#=================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------
#   对于疾病、基因的真实连边和虚拟连边，需要用到新编号字典，对他们进行重新编号

print('=======处理L3关联连边！（重新编号）=========')
#   加载基因编号字典_dict_gene
f = open('./data/Node2vec/_dict_gene.txt', 'r')
_dict_gene = eval(f.read())
f.close()
#   加载疾病编号字典_dict_d
f = open('./data/Node2vec/_dict_d.txt', 'r')
_dict_d = eval(f.read())
f.close()
#   读取基因虚拟连边，并根据新编号映射、保存
data = pd.read_csv('./data/L3/L3_edges.txt')

data['gene1'] = data['gene1'].map(_dict_gene)
data['gene2'] = data['gene2'].map(_dict_gene)

data.to_csv('./data/Node2vec/L3_edges_num.csv', index=False)

#   合并基因的真实连边和虚拟连边，并保存
df1 = pd.read_csv('./data/Node2vec/gene_gene_edges_num.csv')
df2 = pd.read_csv('./data/Node2vec/L3_edges_num.csv')
result_L3 = pd.concat([df1, df2], axis=0)
result_L3.to_csv('./data/Node2vec/gene_gene_L3_edges_num.csv', index=False)

print('L3关联连边重新编号完毕）=========')

#   读取疾病-基因全部连边，并按照新编号映射
data_in = pd.read_csv('./data/Node2vec/DisGeNet_dis_gene.csv')
data_in['Disease'] = data_in['Disease'].map(_dict_d)
data_in['Gene'] = data_in['Gene'].map(_dict_gene)
data_in.to_csv('./data/Node2vec/gene_gene_edges_new_num.csv', index=False)
print('真实关联连边重新编号完毕）=========')

# #-----------------------------------------------------------------------------------------------------------
#对LVR方法预测的连边进行新编号映射，同时产生result_LVR_{}_label_num文件
# 读取原始txt文件
with open('./data/Node2vec/result_3.txt', 'r') as file:
    lines = file.readlines()

# 在列表的开头添加索引
lines.insert(0, 'disease,gene,score\n')

# 将修改后的内容写回文件
with open('./data/Node2vec/result_3.txt', 'w') as file:
    file.writelines(lines)

result_T10 = pd.read_csv('./data/Node2vec/result_3.txt',sep=',')
result_T10['gene'] = result_T10['gene'].map(_dict_gene)
result_T10['disease'] = result_T10['disease'].map(_dict_d)

data1_g = data_in['Gene'].tolist()
data1_d = data_in['Disease'].tolist()
ls_data1 = list(zip(data1_g, data1_d))


_need = result_T10

data_g = _need['gene'].tolist()
data_d = _need['disease'].tolist()
ls_data = list(zip(data_g, data_d))

#   ls_data1是基因-疾病的所有连边，ls_data2是疾病-基因的外部连边，ls_data是LVR方法预测的致病基因与疾病之间的连边
label = []
belong = []
for a in ls_data:
    if a in ls_data1:
        belong.append('in')
        label.append(1)
    else:
        belong.append('neg')
        label.append(0)

_need['label'] = label
_need['belong'] = belong

_need.to_csv('./data/Node2vec/result_LVR_label_num.csv', index=False)

print('=======处理结束！=========')
