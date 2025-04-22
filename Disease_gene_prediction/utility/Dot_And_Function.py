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
import time
import sys
sys.path.append("..")
import os

from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

def sigmoid(z):
    return 1/ (1 + np.exp(-z/100))

#写入日志文件
def write_log(num, epoch, loss, AUC):

    the_time = datetime.datetime.now() #获取当前日期和时间
    
    file_name = '../../results/SAGE/new_feature/new_feature_Double_256_SAGE_top3.txt'

    log_file = open(file_name,'a') #追加模式

    write_value = 'Number:%s, Time:%s, Epoch:%s, Loss:%s, AUC:%s %s' %(num, the_time, epoch, loss, AUC, '\n')

    log_file.write(write_value)

    log_file.close()
#定义求指标的函数-AP、Precision、Recall、F1
#认定得分最高的前i个，预测标签设为1
def get_group_rank(rank_g, top_num):
    if rank_g <= top_num:
        return 1
    else:
        return 0

#定义求AP的算法      
def get_in_AP(hh, i, dis_df, AP_list1, AP_list2):
    #计算AP
    k = hh[hh['disease'] == i].shape[0] #T(d)
    csv = dis_df.copy() #某个疾病的csv文件
    yy = csv.sort_values(by = csv.columns[4])
    top = yy.iloc[:k].copy()
    top['group_rank_pred'] = 1
    
    Pk = top
    AP_list1.append(Pk[Pk['label'] == 1].shape[0])
    AP_list2.append(Pk.shape[0])

    new_AP = sum(AP_list1)/sum(AP_list2)
    
    return new_AP

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


#得到训练集索引、测试集索引、验证机索引、训练集的关联、测试集的关联、验证集的关联
def get_link_edges(eids_dict):
    filename_1 = os.listdir('../../mydata/KFold_10_num/')
    filename_1.pop()
    train_index_10 = []
    test_index_10 = []
    invalid_index_10 = []
    all_index_10 = []

    ls_test_have = [] #存储测试集的所有正样本
    ls_test_no = [] #存储测试集的所有负样本

    for i in filename_1:

        j = i.split('.')[0][7] #得到第i折数据的编号
        #print('-----第：' + j +'折------')
        #得到测试集所有的样本边集合
        data_result = pd.read_csv('../../mydata/Result_10_num/result10_of{}_num.csv'.format(j))
        data_result_dis = data_result['dis'].tolist()
        data_result_gene = data_result['gene'].tolist()
        test_no_edges = list(zip(data_result_gene, data_result_dis))
        #print('测试集里所有的样本集合共有：',len(test_no_edges))

        data = pd.read_csv('../../mydata/KFold_10_num/'+i)

        data1 = data[data['index'] == 'train']#训练集
        data1_gene = data1['gene'].to_list()
        data1_disease = data1['disease'].to_list()
        train_have_edges = list(zip(data1_gene, data1_disease)) #训练集的已连边
        #训练集的索引
        ls_train = []
        for i in train_have_edges:
            ls_train.append(eids_dict[i])

        train_index_10.append(ls_train)

        data2 = data[data['index'] == 'test']#测试集
        data2_gene = data2['gene'].to_list()
        data2_disease = data2['disease'].to_list()#测试集里所有的疾病
        test_have_edges = list(zip(data2_gene, data2_disease)) #测试集的已连边
        ls_test_have.append(test_have_edges)
        #print('测试集里所有的正样本集合共有：',len(test_have_edges))
        #测试集的索引
        ls_test = []
        for i in test_have_edges:
            ls_test.append(eids_dict[i])

        test_index_10.append(ls_test)
        #得到测试集所有的负样本
        test_no_edges = list(set(test_no_edges) - set(test_have_edges))
        #print('测试集里所有的负样本集合共有：',len(test_no_edges))
        ls_test_no.append(test_no_edges)
        #print('负样本集合的长度：',len(ls_test_no))

        data3 = data[data['index'] == 'invalid']#验证集
        data3_gene = data3['gene'].to_list()
        data3_disease = data3['disease'].to_list()#验证集里所有的疾病
        invalid_have_edges = list(zip(data3_gene, data3_disease)) #验证集的已连边

        #验证集的索引
        ls_invalid = []
        for i in invalid_have_edges:
            ls_invalid.append(eids_dict[i])
        invalid_index_10.append(ls_invalid)

        #除去训练集后的所有的索引
        x = ls_test
        #print('x:',len(x))
        y = ls_invalid
        #print('y:',len(y))
        z = x + y
        #print('z:',len(z))
        all_index = z
        all_index_10.append(all_index)
     
    return train_index_10, test_index_10, invalid_index_10, all_index_10, ls_test_have, ls_test_no

# 训练集中的检测结果auc
def DGL_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    scores = F.softmax(scores,dim = 0)
    scores = scores.detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    roc_auc = roc_auc_score(labels, scores)

    return roc_auc
# 间隔损失       
def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean() #1：1

# 测试集中的检测结果auc
def compute_auc(labels, scores):
    roc_auc = roc_auc_score(labels, scores)
    return roc_auc
    
# 测试集中的检测结果auprc
def compute_auprc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision)
    return auprc
    
#得到训练集的图 
def build_train_g(dmhg, all_index, train_index, etype, etype2, eids):
    d,m = dmhg.edges(etype = etype)
    train_g = dgl.remove_edges(dmhg, eids[all_index], etype=etype)
    t_dpeids = dmhg.edge_ids(m[all_index], d[all_index], etype = etype2)
    train_g = dgl.remove_edges(train_g, t_dpeids, etype = etype2)
    
    return train_g
    
#得到训练集的样本 
def train_sample(train_g, etype, no_nodes_with_cat, ls_test_have, ls_test_no, cross_num):
    train_u,train_v = train_g.edges(etype = etype)
    train_have_edges = list(zip(train_u.tolist(), train_v.tolist()))    #已连边
    no_edges = list(set(no_nodes_with_cat)- set(ls_test_no[cross_num]) - set(train_have_edges) - set(ls_test_have[cross_num]))
    train_no_edges = random.sample(no_edges, len(train_have_edges)*1)     #未连边
    
    
    return train_have_edges, train_no_edges
#训练集
def train_f(model, train_g, node_features, train_have_edges, train_no_edges, num_epoch, learn_rate):
    
    model.train()
    opt = torch.optim.Adam(model.parameters(),lr = learn_rate)
    
    for epoch in range(num_epoch):
        # 计算正、负预测分数
        t1 = datetime.datetime.now()
        train_pos_pred = model(train_g, node_features, train_have_edges)
        train_neg_pred = model(train_g, node_features, train_no_edges)
        t2 = datetime.datetime.now()
        # 计算损失
        loss = compute_loss(train_pos_pred, train_neg_pred)
        troc_auc = DGL_auc(train_pos_pred, train_neg_pred)
           
        # 进行反向传播计算
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            print('In epoch {}, loss: {}, auc: {}'.format(epoch, loss, troc_auc))

    
#测试集的特征+分数--拼接
def get_test_score_cat(train_features, test_sample):
    for i in range(len(test_sample)):
            pinjie = torch.cat((train_features['gene'][test_sample[i][0]],train_features['disease'][test_sample[i][1]]),0)
            if i==0:
                input_ = pinjie.unsqueeze(0)
            else:
                input_ = torch.cat((input_, pinjie.unsqueeze(0)), 0)
    
    return input_
    #得到结果文件    
def get_test_result(model, input_, test_sample, test_label):
    df = model.sequence(input_)
    df = df.squeeze(1).tolist()
    
    df = pd.DataFrame(df)
    df.columns = ['pred']

    #df['pred'] = df['pred'].apply(lambda x: sigmoid(x))

    #处理得分文件
    test_pred = np.array(df['pred'].tolist())
    test_pred_ts = torch.tensor(test_pred).unsqueeze(1)

    ts_df = pd.DataFrame(test_sample) #存储测试集所有点对样本
    test_pl = torch.cat((torch.tensor(test_label).unsqueeze(1), test_pred_ts), 1)
    tpl_np = test_pl.detach().numpy()
    tpl_df = pd.DataFrame(tpl_np) #真实标签+预测分数

    result_df = pd.concat((ts_df, tpl_df), 1)

    result_df.columns = ['gene', 'disease', 'label', 'score']

    #对某种疾病和其得分排序
    result_df['rank_g'] = result_df.groupby(['disease'])['score'].rank(ascending = False, method = 'first')
    
    return result_df