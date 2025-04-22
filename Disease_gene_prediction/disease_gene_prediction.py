import datetime
import random
import time

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
import sys
sys.path.append("../../")
from utility.function_set import sigmoid

from utility.function_set import get_group_rank
from utility.function_set import get_in_AP

from utility.function_set import compute_loss
from utility.function_set import DGL_auc
from utility.function_set import compute_auc
from utility.function_set import train_f_new
from utility.function_set import compute_auprc

from utility.function_set import build_train_pos_g
from utility.function_set import build_train_neg_g
from utility.function_set import build_test_neg_g
from utility.function_set import train_sample

from utility.function_set import get_test_result

from utility.GAT_SAGELayer import R_GAT
from utility.GAT_SAGELayer import LinkModel_s_new


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.deterministic = True


# %% md
# 构图
# %%
def getFileColumns(fileName):
    data = pd.read_csv(fileName)
    columns = data.columns
    print(columns)
    column_one = data[columns[0]].to_list()
    column_two = data[columns[1]].to_list()

    return column_one, column_two


# %%
edgesTypeDic = {0: [('gene', 'relate1', 'disease')],
                1: [('disease', 'relate2', 'gene')],
                2: [('gene', 'link', 'gene')]}

etype = ('gene', 'relate1', 'disease')
etype2 = ('disease', 'relate2', 'gene')


def get_graph(etype):
    graph_data = {}

    column_one, column_two = getFileColumns('./Input/dis_gene_edges_new_num.csv')
    graph_data.setdefault(edgesTypeDic[0][0], (column_two, column_one))
    graph_data.setdefault(edgesTypeDic[1][0], (column_one, column_two))

    # 加入L3候选连边
    column_one, column_two = getFileColumns('./Input/gene_gene_all_edges_num.csv')
    graph_data.setdefault(edgesTypeDic[2][0], (column_one + column_two, column_one + column_two))

    disease_gene = pd.read_csv('./Input/dis_gene_edges_new_num.csv')

    hg = dgl.heterograph(graph_data)

    # 将所有的疾病-基因节点连边，存入到nodes_with_cat中，后续用于去重
    disease_ = disease_gene['disease'].to_list()
    gene_ = disease_gene['gene'].to_list()

    nodes_with_cat = list(zip(gene_, disease_))

    # 给边编号
    u, v = hg.edges(etype=etype)
    eids = np.arange(hg.number_of_edges(etype=etype))  # 边编号

    eids_dict = {}

    for xx, _id in enumerate(nodes_with_cat):
        eids_dict.setdefault(_id, xx)

    return hg, eids_dict, eids


# %%
def get_features():
    # 读取基因特征
    features = {}
    data = pd.read_csv('./Input/gene_feature_256_num.csv')
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        gene = tmp[0]
        value = tmp[1:]
        features.setdefault(gene, value)
    matrix = [[0] * 256 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    geneFeat = torch.Tensor(matrix)

    # 疾病特征
    features = {}
    data = pd.read_csv('./Input/dis_feature_256_num.csv')
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        dis = tmp[0]
        value = tmp[1:]
        features.setdefault(dis, value)
    matrix = [[0] * 256 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    diseaseFeat = torch.Tensor(matrix)

    return diseaseFeat, geneFeat


def get_features_LVR():
    # 读取基因特征
    features = {}
    data = pd.read_csv('./Input/gene_feature_num.csv')
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        gene = tmp[0]
        value = tmp[1:]
        features.setdefault(gene, value)
    matrix = [[0] * 128 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    geneFeat = torch.Tensor(matrix)

    # 疾病特征
    features = {}
    data = pd.read_csv('./Input/dis_feature_num.csv')
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        dis = tmp[0]
        value = tmp[1:]
        features.setdefault(dis, value)
    matrix = [[0] * 128 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    diseaseFeat = torch.Tensor(matrix)

    return diseaseFeat, geneFeat


def get_neg_samples():
    data = pd.read_csv('./Input/neg_sample_num.csv')
    no_nodes_with_cat = list(zip(data['gene'].tolist(), data['disease'].tolist()))

    return no_nodes_with_cat, data



setup_seed(13)
hg, eids_dict, eids = get_graph(etype)
g_copy, d_1, d_2 = get_graph(etype)

diseaseFeat, geneFeat = get_features()
hg.nodes['disease'].data['feature'] = diseaseFeat
hg.nodes['gene'].data['feature'] = geneFeat

diseaseFeat1, geneFeat1 = get_features_LVR()
g_copy.nodes['disease'].data['feature'] = diseaseFeat1
g_copy.nodes['gene'].data['feature'] = geneFeat1

no_nodes_with_cat, data = get_neg_samples()

f = open('./Input/_dict_d.txt', 'r')
_dict_d = eval(f.read())
f.close()



target_diseases = pd.read_csv('../data/target_diseases.csv', header=None)
data_dg = pd.read_csv('./Input/dis_gene_edges_new_num.csv')
print(data_dg['gene'].min())
print(data_dg['gene'].max())
print(data_dg['gene'].nunique())


def get_test_edges(disease, dg_train):
    test_tmp = [(x, _dict_d[disease]) for x in range(dg_train['gene'].nunique())]

    train_tmp = dg_train[dg_train['disease'] == _dict_d[each_disease]]

    train_tmp_d = train_tmp['disease'].tolist()
    train_tmp_g = train_tmp['gene'].tolist()
    node_tmp = list(zip(train_tmp_g, train_tmp_d))

    test_node = list(set(test_tmp) - set(node_tmp))

    edges_df = pd.DataFrame(test_node)
    edges_df.columns = ['gene', 'disease']

    _dis = edges_df['disease'].tolist()
    _gene = edges_df['gene'].tolist()
    test_no = list(zip(_gene, _dis))

    return test_no


test_no_edges = []

for each_disease in target_diseases[0]:
    test_no_edges = test_no_edges + get_test_edges(each_disease, data_dg)


# data1 = data_dg[data_dg['index'] != 'case']
data1 = data_dg
data1_gene = data1['gene'].to_list()
data1_disease = data1['disease'].to_list()
train_have_edges = list(zip(data1_gene, data1_disease))

train_index = []
for a in train_have_edges:
    train_index.append(eids_dict[a])

print(len(train_index))


train_pos_g, test_pos_g = build_train_pos_g(hg, train_index, [], [], etype, etype2, eids)
train_u, train_v = train_pos_g.edges(etype=etype)
train_have_edges = list(zip(train_u.tolist(), train_v.tolist()))

train_no_edges = random.sample(no_nodes_with_cat, len(train_have_edges) * 50)
train_neg_g = build_train_neg_g(hg, train_no_edges, train_index, [], etype, etype2, eids)

rel_names = ['relate1', 'relate2', 'link']

model = LinkModel_s_new(256, 128, 64, 8, rel_names)

disease_feats = hg.nodes['disease'].data['feature']
gene_feats = hg.nodes['gene'].data['feature']

disease_feats1 = g_copy.nodes['disease'].data['feature']
gene_feats1 = g_copy.nodes['gene'].data['feature']

node_features = {'gene': gene_feats, 'disease': disease_feats}
LVR_f = {'gene': gene_feats1, 'disease': disease_feats1}

train_f_new(model, train_pos_g, train_neg_g, node_features, LVR_f, 10, 0.0009, etype)

model.eval()
with torch.no_grad():
    train_features = model.rgat(train_pos_g, node_features)

    test_label = []
    test_belong = []

    for y in range(len(test_no_edges)):
        test_label.append(0)
        test_belong.append('neg')

    test_neg_g = build_test_neg_g(hg, test_no_edges, train_index, [], etype, etype2, eids)

    test_pos_score = model.pred(test_pos_g, train_features, LVR_f, etype)
    test_neg_score = model.pred(test_neg_g, train_features, LVR_f, etype)

    scores = test_pos_score.tolist() + test_neg_score.tolist()
    result_df = get_test_result(test_pos_g, test_neg_g, test_pos_score, test_neg_score, test_label, etype)

result_df = result_df.sort_values(by=['disease', 'rank_g'])

f = open('./Input/_dict_gene.txt', 'r')
_dict_gene = eval(f.read())
f.close()

re_dict_gene = {value: key for key, value in _dict_gene.items()}
re_dict_d = {value: key for key, value in _dict_d.items()}

for each_disease in list(target_diseases[0]):
    df_tmp = result_df[result_df['disease'] == _dict_d[each_disease]]

    df_tmp['disease'] = df_tmp['disease'].map(re_dict_d)
    df_tmp['gene'] = df_tmp['gene'].map(re_dict_gene)
    df_tmp[df_tmp['rank_g'] <= 20].to_csv('Output/' + each_disease + '.csv', index=False)

dis_708 = result_df[result_df['disease'] == 705].copy()

dis_708['disease'] = dis_708['disease'].map(re_dict_d)
dis_708['gene'] = dis_708['gene'].map(re_dict_gene)

print(dis_708[dis_708['rank_g'] <= 20])