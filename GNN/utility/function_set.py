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

from tqdm import tqdm

import numpy as np
import datetime
import sys
sys.path.append("..")
import os


from sklearn.metrics import roc_auc_score, average_precision_score

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

#构图
def getFileColumns(fileName):
    data = pd.read_csv(fileName)
    columns = data.columns
    column_one = data[columns[0]].to_list()
    column_two = data[columns[1]].to_list()
    
    return column_one, column_two

def get_graph(fileName_dg_num, fileName_ggL3_num, edgesTypeDic, etype):
    graph_data = {}

    column_one, column_two = getFileColumns(fileName_dg_num)
    graph_data.setdefault(edgesTypeDic[0][0], (column_two, column_one))
    graph_data.setdefault(edgesTypeDic[1][0], (column_one, column_two))
    
    #加入L3候选连边
    column_one, column_two = getFileColumns(fileName_ggL3_num)
    graph_data.setdefault(edgesTypeDic[2][0], (column_one+column_two, column_one+column_two))

    disease_gene = pd.read_csv(fileName_dg_num)


    hg = dgl.heterograph(graph_data)  

    #130820条疾病-基因节点连边
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

#获取节点特征
def get_features(feature_gene256, feature_dis256):
    #读取基因特征
    features = {}
    data = pd.read_csv(feature_gene256)
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        gene = tmp[0]
        value = tmp[1:]
        features.setdefault(gene, value)
    matrix = [[0]*256 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    geneFeat = torch.Tensor(matrix)

    #疾病特征
    features = {}
    data = pd.read_csv(feature_dis256)
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        dis = tmp[0]
        value = tmp[1:]
        features.setdefault(dis, value)
    matrix = [[0]*256 for _ in range (len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    diseaseFeat = torch.Tensor(matrix)
    
    return diseaseFeat, geneFeat

#获取节点初始特征-LVR
def get_features_LVR(LVR_gene128, LVR_dis128):
    #读取基因特征
    features = {}
    data = pd.read_csv(LVR_gene128)
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        gene = tmp[0]
        value = tmp[1:]
        features.setdefault(gene, value)
    matrix = [[0]*128 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    geneFeat = torch.Tensor(matrix)

    #疾病特征
    features = {}
    data = pd.read_csv(LVR_dis128)
    for aa in range(len(data)):
        tmp = list(data.loc[aa])
        dis = tmp[0]
        value = tmp[1:]
        features.setdefault(dis, value)
    matrix = [[0]*128 for _ in range(len(data))]
    for j in range(len(data)):
        value = features[j]
        matrix[j] = value
    diseaseFeat = torch.Tensor(matrix)
    
    return diseaseFeat, geneFeat

#获取负样本
def get_neg_samples(neg_sample):
    data = pd.read_csv(neg_sample)
    no_nodes_with_cat = list(zip(data['gene'].tolist(), data['disease'].tolist()))
    
    return no_nodes_with_cat, data

# 获取LVR的前Top-k生成的结果文件
def get_test_sample(LVR_result_gene):
    
    data_test = pd.read_csv(LVR_result_gene)
    
    # 获取原预测文件给定的样本
    _label1 = data_test[(data_test['label'] == 1) & (data_test['belong'] != 'ex')]
    _dis = _label1['disease'].tolist()
    _gene = _label1['gene'].tolist()
    test_have_edges = list(zip(_gene,_dis)) #测试集中的已连边
    
    _label0 = data_test[data_test['label'] == 0]
    _dis = _label0['disease'].tolist()
    _gene = _label0['gene'].tolist()
    test_no = list(zip(_gene,_dis)) #测试集中的未连边
    
    _ex = data_test[(data_test['label'] == 1) & (data_test['belong'] == 'ex')]
    _dis = _ex['disease'].tolist()
    _gene = _ex['gene'].tolist()
    test_ex_edges = list(zip(_gene,_dis)) #测试集中涉及到的外部连边
    
    test_no_edges = test_no + test_ex_edges
    
    return test_have_edges, test_no, test_ex_edges, test_no_edges

# 读取连边文件并获取对应的边ID,以便后续构图
def get_edge_ID(fileName_dg_num, test_have_edges, eids_dict):
    
    file_KF = pd.read_csv(fileName_dg_num)
    
    #训练集的所有连边
    data1 = file_KF[file_KF['index'] == 'train']
    data1_gene = data1['gene'].to_list()
    data1_disease = data1['disease'].to_list()
    train_have_edges = list(zip(data1_gene, data1_disease))
    #测试集的所有连边
    data2 = file_KF[file_KF['index'] == 'test']
    data2_gene = data2['gene'].to_list()
    data2_disease = data2['disease'].to_list()#测试集里所有的疾病
    test_all = list(zip(data2_gene, data2_disease)) #测试集的已连边
    
    #得到对应的索引号以便去除连边
    #训练集的索引
    train_index = []
    for a in train_have_edges:
        train_index.append(eids_dict[a])

    #测试集的索引
    test_index = []
    for b in test_all:
        test_index.append(eids_dict[b])

    test_index_HerPred = [] #该索引用于建立测试集的正样本图
    for c in test_have_edges:
        test_index_HerPred.append(eids_dict[c])
        
    return train_index, test_index, test_index_HerPred, data2

# 得到测试集中的标签
def get_test_belong(test_have_edges, test_no, test_ex_edges):
    test_label = []
    test_belong = []
    for x in range(len(test_have_edges)):
        test_label.append(1)
        test_belong.append('in')

    for y in range(len(test_no)):
        test_label.append(0)
        test_belong.append('neg')

    for z in range(len(test_ex_edges)):
        test_label.append(1)
        test_belong.append('ex')
    
    return test_label, test_belong

# 评估指标
def get_AP_pre_rec_auc_auprc(df, test):
    
    df['rank_'] = df[['disease','score']].groupby('disease').rank(method='dense',ascending=False)
    df = df[df['rank_']<=30].reset_index(drop=True)

    test_ = test[test['index']=='test'].reset_index(drop=True)
    test_sta = test_[['disease','gene']].groupby('disease',as_index=False).agg(list)
    test_sta_dict = dict(zip(test_sta['disease'],test_sta['gene']))

    disease_list = list(df['disease'].unique())

    ap_list_1 = []
    ap_list_2 = []

    for each_dis in tqdm(disease_list):
        df_tmp = df[df['disease']==each_dis]
        df_tmp['rank'] = df_tmp['score'].rank(method='dense',ascending=False)
        len_Td = min(len(test_sta_dict[each_dis]),10)

        df_tmp = df_tmp[df_tmp['rank']<=len_Td].reset_index(drop=True)

        ap_list_1.append(sum((df_tmp['label']==1) & (df_tmp['belong']=='in')))
        ap_list_2.append(len_Td)


    ap_tmp = sum(ap_list_1)/sum(ap_list_2)
    
    
    top_k = 10
    pre_list_10 = []
    recall_list_10 = []
    f1_list_10 = []

    for each_dis in tqdm(disease_list):
        df_tmp = df[df['disease']==each_dis]
        df_tmp['rank'] = df_tmp['score'].rank(method='dense',ascending=False)
        df_tmp = df_tmp[df_tmp['rank']<=top_k].reset_index(drop=True)

        len_Td = len(test_sta_dict[each_dis])

        pre_tmp = sum((df_tmp['label']==1) & (df_tmp['belong']=='in'))/top_k
        pre_list_10.append(pre_tmp)

        recall_tmp = sum((df_tmp['label']==1) & (df_tmp['belong']=='in'))/len_Td
        recall_list_10.append(recall_tmp)

        f1_tmp = 2*sum((df_tmp['label']==1) & (df_tmp['belong']=='in'))/(top_k+len_Td)
        f1_list_10.append(f1_tmp)
        
    top_k = 3
    pre_list_3 = []
    recall_list_3 = []
    f1_list_3 = []

    for each_dis in tqdm(disease_list):
        df_tmp = df[df['disease']==each_dis]
        df_tmp['rank'] = df_tmp['score'].rank(method='dense',ascending=False)
        df_tmp = df_tmp[df_tmp['rank']<=top_k].reset_index(drop=True)


        len_Td = len(test_sta_dict[each_dis])

        pre_tmp = sum((df_tmp['label']==1) & (df_tmp['belong']=='in'))/top_k
        pre_list_3.append(pre_tmp)

        recall_tmp = sum((df_tmp['label']==1) & (df_tmp['belong']=='in'))/len_Td
        recall_list_3.append(recall_tmp)

        f1_tmp = 2*sum((df_tmp['label']==1) & (df_tmp['belong']=='in'))/(top_k+len_Td)
        f1_list_3.append(f1_tmp)
        
    auc_list = []
    auprc_list = []

    for each_dis in tqdm(disease_list):
        df_tmp = df[df['disease']==each_dis]

        try:
            auc_tmp = roc_auc_score(df_tmp['label'],df_tmp['score'])
            auc_list.append(auc_tmp)

            auprc_tmp = average_precision_score(df_tmp['label'],df_tmp['score'])
            auprc_list.append(auprc_tmp)
        except:
#             auc_list.append(0)
#             auprc_list.append(0)
            pass
    
    return ap_tmp, pre_list_10, recall_list_10, f1_list_10, pre_list_3, recall_list_3, f1_list_3, auc_list, auprc_list

#认定得分最高的前i个，预测标签设为1
def get_group_rank(rank_g, top_num):
    if rank_g <= top_num:
        return 1
    else:
        return 0

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
