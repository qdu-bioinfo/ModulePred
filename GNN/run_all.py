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

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("././")
from utility.function_set import sigmoid
from utility.function_set import getFileColumns
from utility.function_set import get_graph
from utility.function_set import get_features
from utility.function_set import get_features_LVR
from utility.function_set import get_neg_samples
from utility.function_set import get_test_sample
from utility.function_set import get_edge_ID
from utility.function_set import get_test_belong
from utility.function_set import get_AP_pre_rec_auc_auprc

from utility.function_set import get_group_rank

from utility.function_set import compute_loss
from utility.function_set import DGL_auc
from utility.function_set import train_f_new

from utility.function_set import build_train_pos_g
from utility.function_set import build_train_neg_g
from utility.function_set import build_test_neg_g
from utility.function_set import train_sample

from utility.function_set import get_test_result

from utility.GAT_SAGELayer import R_GAT
from utility.GAT_SAGELayer import LinkModel_s_new


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.deterministic = True

edgesTypeDic = {0: [('gene', 'relate1', 'disease')],
                1: [('disease', 'relate2', 'gene')],
                2: [('gene', 'link', 'gene')]}

etype = ('gene', 'relate1', 'disease')
etype2 = ('disease', 'relate2', 'gene')

pre_top10_list = []
rec_top10_list = []
f1_top10_list = []

pre_top3_list = []
rec_top3_list = []
f1_top3_list = []

ap_list_cv = []
auc_list_cv = []
auprc_list_cv = []

times = 0

# 设置随机数种子
setup_seed(13)

fileName_dg_num = './data/GNN/dis_gene_edges_num.csv'
fileName_ggL3_num = './data/GNN/gene_gene_L3_edges_num.csv'

feature_gene256 = './data/GNN/gene_feature_256_num.csv'
feature_dis256 = './data/GNN/dis_feature_256_num.csv'

LVR_gene128 = './data/GNN/gene_feature_num.csv'
LVR_dis128 = './data/GNN/dis_feature_num.csv'

neg_sample = './data/GNN/neg_sample_num.csv'

LVR_result_gene = './data/GNN/result_LVR_label_num.csv'

times += 1
print('----------------------------第', times, '次----------------------------')

print('==构建异构生物分子网络......==')
hg, eids_dict, eids = get_graph(fileName_dg_num, fileName_ggL3_num, edgesTypeDic, etype)
g_copy, d_1, d_2 = get_graph(fileName_dg_num, fileName_ggL3_num, edgesTypeDic, etype)

print('==传入特征......')

diseaseFeat, geneFeat = get_features(feature_gene256, feature_dis256)
hg.nodes['disease'].data['feature'] = diseaseFeat
hg.nodes['gene'].data['feature'] = geneFeat

diseaseFeat1, geneFeat1 = get_features_LVR(LVR_gene128, LVR_dis128)
g_copy.nodes['disease'].data['feature'] = diseaseFeat1
g_copy.nodes['gene'].data['feature'] = geneFeat1

print('==获取所有负样本......')
no_nodes_with_cat, data = get_neg_samples(neg_sample)

print('==获取所有测试集样本......')
test_have_edges, test_no, test_ex_edges, test_no_edges = get_test_sample(LVR_result_gene)

print('==获取连边ID......')
train_index, test_index, test_index_HerPred, test = get_edge_ID(fileName_dg_num, test_have_edges, eids_dict)

print('==得到训练集的正样本图和测试集的正样本图......')
train_pos_g, test_pos_g = build_train_pos_g(hg, train_index, test_index, test_index_HerPred, etype, etype2, eids)

print('==得到训练集中的samples......')
train_u, train_v = train_pos_g.edges(etype=etype)
train_have_edges = list(zip(train_u.tolist(), train_v.tolist()))  # 已连边
train_no_edges = random.sample(list(set(no_nodes_with_cat) - set(test_no_edges) - set(test_ex_edges)),
                                   len(train_have_edges) * 1)  # 未连边

print('==得到训练集的负样本图......')
train_neg_g = build_train_neg_g(hg, train_no_edges, train_index, test_index, etype, etype2, eids)

rel_names = ['relate1', 'relate2', 'link']

model = LinkModel_s_new(256, 128, 64, 8, rel_names)

disease_feats = hg.nodes['disease'].data['feature']
gene_feats = hg.nodes['gene'].data['feature']

disease_feats1 = g_copy.nodes['disease'].data['feature']
gene_feats1 = g_copy.nodes['gene'].data['feature']

node_features = {'gene': gene_feats, 'disease': disease_feats}
LVR_f = {'gene': gene_feats1, 'disease': disease_feats1}

print('==开始训练：')
train_f_new(model, train_pos_g, train_neg_g, node_features, LVR_f, 10, 0.0009, etype)

print('==开始测试：')

model.eval()
with torch.no_grad():
    train_features = model.rgat(train_pos_g, node_features)

    print('对疾病与其他节点关联预测的评分：')
#     test_label, test_belong = get_test_belong(test_have_edges, test_no, test_ex_edges)

#     print('==得到测试集的负样本图及分数......')
#     未连边，连边 ，空的
    test_no_edges = [(4, 0), (5, 0), (6, 0), (7, 0), (8, 0)]
    train_index = [0, 1, 2, 3]
    test_index = []
    test_neg_g = build_test_neg_g(hg, test_no_edges, train_index, test_index, etype, etype2, eids)
#     test_pos_score = model.pred(test_pos_g, train_features, LVR_f, etype)
    test_neg_score = model.pred(test_neg_g, train_features, LVR_f, etype)
    df = pd.DataFrame({
        'disease': ['d0', 'd0', 'd0', 'd0', 'd0'],
        'gene': ['g5', 'g6', 'g7', 'g8', 'g9'],
        'score': test_neg_score.view(-1).numpy()  # Convert tensor to a 1D numpy array
    })
    print(df)
    df.to_csv('./data/GNN/score.csv')
#     scores = test_pos_score.tolist() + test_neg_score.tolist()
#
#     print('==得到测试集的结果文件......')
#     df = get_test_result(test_pos_g, test_neg_g, test_pos_score, test_neg_score, test_label, etype)
#
#     df['group_rank_pred'] = df.apply(lambda x: get_group_rank(x.rank_g, 10), axis=1)
#     df['belong'] = test_belong
#     # result_df.to_csv('results/SAGE/ModPred/result_{}_mod.csv'.format(i), index = False)
#
#     print('==评估.......')
#     ap_tmp, pre_list_10, recall_list_10, f1_list_10, pre_list_3, recall_list_3, f1_list_3, auc_list, auprc_list = get_AP_pre_rec_auc_auprc(
#         df, test)
#
#     ap_list_cv.append(ap_tmp)
#     print('AP: ' + str(ap_tmp))
#
#     pre = np.mean(pre_list_10)
#     recall = np.mean(recall_list_10)
#     f1 = np.mean(f1_list_10)
#     print('top10_Pre: ' + str(pre))
#     print('top10_Recall: ' + str(recall))
#     print('top10_F1: ' + str(f1))
#
#     pre_top10_list.append(pre)
#     rec_top10_list.append(recall)
#     f1_top10_list.append(f1)
#
#     pre = np.mean(pre_list_3)
#     recall = np.mean(recall_list_3)
#     f1 = np.mean(f1_list_3)
#     print('top3_Pre: ' + str(pre))
#     print('top3_Recall: ' + str(recall))
#     print('top3_F1: ' + str(f1))
#
#     pre_top3_list.append(pre)
#     rec_top3_list.append(recall)
#     f1_top3_list.append(f1)
#
#     auc_list_cv.append(np.mean(auc_list))
#     auprc_list_cv.append(np.mean(auprc_list))
#     print('AUC: ' + str(np.mean(auc_list)))
#     print('AUPRC: ' + str(np.mean(auprc_list)))
# print('================================')
# print('10折平均AP:', np.mean(ap_list_cv))
#
# print('10折平均Precision_Top-3:', np.mean(pre_top3_list))
# print('10折平均Recall_Top-3:', np.mean(rec_top3_list))
# print('10折平均F1-score_Top-3:', np.mean(f1_top3_list))
#
# print('10折平均Precision_Top-10:', np.mean(pre_top10_list))
# print('10折平均Recall_Top-10:', np.mean(rec_top10_list))
# print('10折平均F1-score_Top-10:', np.mean(f1_top10_list))
#
# print('10折平均AUC:', np.mean(auc_list_cv))
# print('10折平均AUPRC:', np.mean(auprc_list_cv))