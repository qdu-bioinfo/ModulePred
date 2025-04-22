import dgl
import dgl.nn as dglnn
import dgl.function as fn

import pandas as pd
import random

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

import numpy as np
import datetime
import sys
sys.path.append("..")
import os

from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

#定义求指标的函数-AP、Precision、Recall、F1
#认定得分最高的前i个，预测标签设为1
def get_group_rank(rank_g, top_num):
    if rank_g <= top_num:
        return 1
    else:
        return 0

#定义求AP的算法-原论文   
def get_in_AP(hh, i, dis_df):
    #计算AP
    k = hh[hh['disease'] == i].shape[0] #T(d)
    csv = dis_df.copy() #某个疾病的csv文件
    Pk= csv[csv['rank_g'] <= k]
    
    return Pk[Pk['label'] == 1].shape[0], k

#定义求Precision、Recall和F1分数的方法
def get_hu_PRF(hh, i, dis_df, top_i, the_type):
    if the_type == 'in':
        #计算precision
        csv = dis_df.copy() #某个疾病的csv文件
        yy = csv.sort_values(by = csv.columns[4])
        top = yy.iloc[:top_i].copy()
        top['group_rank_pred'] = 1

        prec = top

        prec_true = prec[prec['label'] == prec['group_rank_pred']].shape[0]#预测正确的个数
        if prec.shape[0] == 0: 
            dis_precision = 0

        if prec.shape[0] != 0:
            dis_precision = prec_true/prec.shape[0]

        #计算recall
        all_true = hh[hh['disease'] == i].shape[0]#针对于i号疾病的真实连边的个数

        reca_true = prec_true#预测正确的个数

        if all_true == 0: 
            dis_recall = 0

        else:
            dis_recall = reca_true/all_true

        #计算F1
        if dis_precision == 0 or dis_recall == 0:
            dis_f1 = 0

        else:
            dis_f1 = 2*dis_precision*dis_recall/(dis_precision + dis_recall)

        return dis_precision, dis_recall, dis_f1
    
    if the_type == 'external':
        #计算precision
        prec = dis_df[dis_df['rank_g'] <= top_i]
        prec_true = prec[prec['label'] == prec['group_rank_pred']].shape[0]#预测正确的个数
        if prec.shape[0] == 0: 
            dis_precision = 0

        if prec.shape[0] != 0:
            dis_precision = prec_true/prec.shape[0]

        #计算recall
        all_true = hh[hh['disease'] == i].shape[0]#针对于i号疾病的真实连边的个数

        reca = dis_df[dis_df['rank_g'] <= top_i]

        reca_true = reca[reca['label'] == reca['group_rank_pred']].shape[0]#预测正确的个数

        if all_true == 0: 
            dis_recall = 0

        else:
            dis_recall = reca_true/all_true

        #计算F1
        if dis_precision == 0 or dis_recall == 0:
            dis_f1 = 0

        else:
            dis_f1 = 2*dis_precision*dis_recall/(dis_precision + dis_recall)

        return dis_precision, dis_recall, dis_f1


# 训练集中的检测结果auc
def DGL_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    #scores = F.softmax(scores,dim = 0)
    scores = scores.detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    roc_auc = roc_auc_score(labels, scores)

    return roc_auc
# 间隔损失       
def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()

# 测试集中的检测结果auc
def compute_auc(labels, scores):
    roc_auc = roc_auc_score(labels, scores)
    return roc_auc
    
# 测试集中的检测结果auprc
def compute_auprc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision)
    return auprc
    
#得到训练集的正样本图 
def build_train_pos_g(dmhg, train_index, test_index, test_index_HerPred, etype, etype2, eids):
    #得到训练集的正样本图
    d,m = dmhg.edges(etype = etype)
    train_g = dgl.remove_edges(dmhg, eids[test_index], etype=etype)
    t_dpeids = dmhg.edge_ids(m[test_index], d[test_index], etype = etype2)
    train_pos_g = dgl.remove_edges(train_g, t_dpeids, etype = etype2)
    
    #得到测试集的正样本图
    index_test = list(set(test_index) - set(test_index_HerPred))
    
    test_g = dgl.remove_edges(dmhg, eids[np.array(train_index + index_test)], etype=etype)
    t_dpeids = dmhg.edge_ids(m[np.array(train_index + index_test)], d[np.array(train_index + index_test)], etype = etype2)
    test_pos_g = dgl.remove_edges(test_g, t_dpeids, etype = etype2)
    
    return train_pos_g, test_pos_g
    
#得到训练集的正样本图_1 
def build_train_pos_g_all(dmhg, train_index, test_index, etype, etype2, eids):
    #得到训练集的正样本图
    d,m = dmhg.edges(etype = etype)
    train_g = dgl.remove_edges(dmhg, eids[test_index], etype=etype)
    t_dpeids = dmhg.edge_ids(m[test_index], d[test_index], etype = etype2)
    train_pos_g = dgl.remove_edges(train_g, t_dpeids, etype = etype2)
    
    #得到测试集的正样本图
    d,m = dmhg.edges(etype = etype)
    test_g = dgl.remove_edges(dmhg, eids[train_index], etype=etype)
    t_dpeids = dmhg.edge_ids(m[train_index], d[train_index], etype = etype2)
    test_pos_g = dgl.remove_edges(test_g, t_dpeids, etype = etype2)
    
    return train_pos_g, test_pos_g
    
    
#得到训练集的正负样本边
def train_sample(train_g, etype, no_nodes_with_cat, ls_test_have, ls_test_no, cross_num):
    train_u,train_v = train_g.edges(etype = etype)
    train_have_edges = list(zip(train_u.tolist(), train_v.tolist()))    #已连边
    no_edges = list(set(no_nodes_with_cat)- set(ls_test_no[cross_num]) - set(train_have_edges) - set(ls_test_have[cross_num]))
    train_no_edges = random.sample(no_edges, len(train_have_edges)*1)     #未连边
    
    
    return train_have_edges, train_no_edges
#得到训练集的负样本图 
def build_train_neg_g(hg, train_no_edges, train_index, test_index, etype, etype2, eids):
    
    data1 = pd.DataFrame(train_no_edges)
    data1.columns = ['gene', 'disease']

    #得到负样本图
    #1.移除所有的已连边D-G
    d,m = hg.edges(etype = etype)
    g = dgl.remove_edges(hg, eids[np.array(train_index + test_index)], etype=etype)
    t_dpeids = hg.edge_ids(m[np.array(train_index + test_index)], d[np.array(train_index + test_index)], etype = etype2)
    train_neg_g = dgl.remove_edges(g, t_dpeids, etype = etype2)
    #2.添加负样本连边信息
    train_neg_g.add_edges(np.array(data1['gene']), np.array(data1['disease']), etype='relate1')
    train_neg_g.add_edges(np.array(data1['disease']), np.array(data1['gene']), etype='relate2')
    #得到负样本图
    return train_neg_g

#训练集
def train_f(model, train_pos_g, train_neg_g, node_features, num_epoch, learn_rate, etype):
    
    model.train()
    opt = torch.optim.Adam(model.parameters(),lr = learn_rate)
    
    for epoch in range(num_epoch):
        # 计算正、负预测分数
        train_pos_pred, train_neg_pred = model(train_pos_g, train_neg_g, node_features, etype)

        # 计算损失
        loss = compute_loss(train_pos_pred, train_neg_pred)
        troc_auc = DGL_auc(train_pos_pred, train_neg_pred)
           
        # 进行反向传播计算
        opt.zero_grad()
        loss.backward()
        opt.step()
        #if epoch % 10 == 0:
        print('In epoch {}, loss: {}, auc: {}'.format(epoch, loss, troc_auc))

#训练集
def train_f_new(model, train_pos_g, train_neg_g, node_features, LVR_f, num_epoch, learn_rate, etype):
    
    model.train()
    opt = torch.optim.Adam(model.parameters(),lr = learn_rate)
    
    for epoch in range(num_epoch):
        # 计算正、负预测分数
        train_pos_pred, train_neg_pred = model(train_pos_g, train_neg_g, node_features, LVR_f, etype)

        # 计算损失
        loss = compute_loss(train_pos_pred, train_neg_pred)
        troc_auc = DGL_auc(train_pos_pred, train_neg_pred)
           
        # 进行反向传播计算
        opt.zero_grad()
        loss.backward()
        opt.step()
        #if epoch % 10 == 0:
        print('In epoch {}, loss: {}, auc: {}'.format(epoch, loss, troc_auc))
        
#得到测试集的负样本图 
def build_test_neg_g(hg, test_no_edges, train_index, test_index, etype, etype2, eids):
    
    data1 = pd.DataFrame(test_no_edges)
    data1.columns = ['gene', 'disease']

    #得到负样本图
    #1.移除所有的已连边D-G
    d,m = hg.edges(etype = etype)
    g = dgl.remove_edges(hg, eids[np.array(train_index + test_index)], etype=etype)
    t_dpeids = hg.edge_ids(m[np.array(train_index + test_index)], d[np.array(train_index + test_index)], etype = etype2)
    test_neg_g = dgl.remove_edges(g, t_dpeids, etype = etype2)
    #2.添加负样本连边信息
    test_neg_g.add_edges(np.array(data1['gene'].tolist()), np.array(data1['disease'].tolist()), etype='relate1')
    test_neg_g.add_edges(np.array(data1['disease'].tolist()), np.array(data1['gene'].tolist()), etype='relate2')
    #得到负样本图
    return test_neg_g
        #测试集
    
#测试集的特征+分数--拼接
def get_test_score_cat(train_features, test_sample):
    for i in range(len(test_sample)):
            pinjie = torch.cat((train_features['gene'][test_sample[i][0]],train_features['disease'][test_sample[i][1]]),0)
            if i==0:
                input_ = pinjie.unsqueeze(0)
            else:
                input_ = torch.cat((input_, pinjie.unsqueeze(0)), 0)
    
    df = pd.DataFrame(np.array(input_))
    
    return df
    #得到结果文件

def get_test_result(test_pos_g, test_neg_g, test_pos_score, test_neg_score, test_label, etype):
    src, dst = test_pos_g.edges(etype=etype)
    src1, dst1 = test_neg_g.edges(etype=etype)
    
    src = src.cpu().numpy()  # 将源节点转换为NumPy数组
    dst = dst.cpu().numpy()  # 将目标节点转换为NumPy数组

    src1 = src1.cpu().numpy()  # 将源节点转换为NumPy数组
    dst1 = dst1.cpu().numpy()  # 将目标节点转换为NumPy数组
    
    result_df = pd.DataFrame(list(zip(src.tolist() + src1.tolist(), dst.tolist() + dst1.tolist())))
    result_df.columns = ['gene', 'disease']
    
    result_df['score'] = test_pos_score.squeeze(1).tolist() + test_neg_score.squeeze(1).tolist()
    
    result_df['label'] = test_label

    #对某种疾病和其得分排序
    result_df['rank_g'] = result_df.groupby(['disease'])['score'].rank(ascending = False, method = 'first')
    
    return result_df

def getFileColumns(fileName):
    data = pd.read_csv(fileName)
    columns = data.columns
    print(columns)
    column_one = data[columns[0]].to_list()
    column_two = data[columns[1]].to_list()
    
    return column_one, column_two

# 构建异构生物分子网络图
def get_graph(i, etype, edgesTypeDic):
    graph_data = {}

    column_one, column_two = getFileColumns('data/DGN/KFold_{}/dis_gene_edges_{}_num.csv'.format(i, i))
    graph_data.setdefault(edgesTypeDic[0][0], (column_two, column_one))
    graph_data.setdefault(edgesTypeDic[1][0], (column_one, column_two))
    
    #加入L3候选连边
    column_one, column_two = getFileColumns('data/DGN/KFold_{}/gene_gene_L3_edges_{}_num.csv'.format(i, i))
    graph_data.setdefault(edgesTypeDic[2][0], (column_one+column_two, column_one+column_two))

    disease_gene = pd.read_csv('data/DGN/KFold_{}/dis_gene_edges_{}_num.csv'.format(i, i))


    hg = dgl.heterograph(graph_data)  

    #将所有的疾病-基因节点连边，存入到nodes_with_cat中，后续用于去重
    disease_ = disease_gene['disease'].to_list()
    gene_ = disease_gene['gene'].to_list()

    nodes_with_cat = list(zip(gene_, disease_))

    #给边编号
    u,v = hg.edges(etype = etype)
    eids = np.arange(hg.number_of_edges(etype = etype)) # 边编号

    eids_dict = {}

    for xx, _id in enumerate(nodes_with_cat):
        eids_dict.setdefault(_id, xx)
        
    return hg, eids_dict, eids
# 读取特征
def get_features(i, num):
    #读取基因特征
    features = {}
    data = pd.read_csv('data/DGN/KFold_{}/gene_feature_{}_{}_num.csv'.format(i, num, i))
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        gene = tmp[0]
        value = tmp[1:]
        features.setdefault(gene, value)
    matrix = [[0]*192 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    geneFeat = torch.Tensor(matrix)

    #疾病特征
    features = {}
    data = pd.read_csv('data/DGN/KFold_{}/dis_feature_{}_{}_num.csv'.format(i, num, i))
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        dis = tmp[0]
        value = tmp[1:]
        features.setdefault(dis, value)
    matrix = [[0]*192 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    diseaseFeat = torch.Tensor(matrix)
    
    return diseaseFeat, geneFeat

#获取负样本
def get_neg_samples(i):
    data = pd.read_csv('data/DGN/KFold_{}/neg_sample_num_{}.csv'.format(i, i))
    no_nodes_with_cat = list(zip(data['gene'].tolist(), data['disease'].tolist()))
    
    return no_nodes_with_cat, data