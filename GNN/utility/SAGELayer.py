import torch.nn as nn
import torch.nn.functional as F
import torch
import torch as th
from torch import nn
from torch.nn import init

import numpy as np
import pandas as pd

import dgl
import dgl.nn as dglnn
import dgl.function as fn

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.transforms import reverse
from dgl.convert import block_to_graph
from dgl.heterograph import DGLBlock

from utility.GCNLayer import GCNlayer
from utility.GCNLayer import HereroCatPredictor

class sageconv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(sageconv, self).__init__()
        valid_aggre_types = {'mean', 'gcn', 'pool', 'lstm'}#gcn 聚合可以理解为周围所有的邻居结合和当前节点的均值
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )
            
        #调用expand_as_pair,如果in_feats是tuple直接返回
        
        #如果in_feats是int,则返回两相同此int值，分别表示源、目标节点特征维度(同构图情形）
        
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        
        #aggregator type:mean/pool/lstm/gcn
        
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()
 
    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):

        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def _lstm_reducer(self, nodes):
        
        # mailbox['m']返回batch_size,seq_len,dimension，因此__init__中LSTM定义中batch_first=True

        m = nodes.mailbox['m'] #用于暂存消息函数发送过来的数据。
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}


#1.消息产生和传递 2.消息聚合 3.输出更新后的特征
    def forward(self, graph, feat, edge_weight=None):

        self._compatibility_check()
        with graph.local_scope():
            
            #抽取起点特征、终点特征，再加dropout.
            
            if isinstance(feat, tuple):          #单二分图
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)       #同构图
                if graph.is_block:  #子图在DGL中称为区块(block)
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    
                    #在区块创建的阶段，dst nodes 位于节点列表的最前面。通过索引 [0:g.number_of_dst_nodes()] 可以找到 feat_dst
                    
            #此为消息函数，把起点的h特征拷贝到边的m特征上。
            
            msg_fn = fn.copy_u('h', 'm')
            
            #如果边有权重，则调用内置u_mul_e,意为把起点的h特征乘以边权重，再将结果赋给边的m特征。
            
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            #提前记录目标节点的原始特征
            
            h_self = feat_dst

            # Handle the case of graphs without edges
            
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # 检查起点特征维度与输出特征维度。
            
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            
            #1.mean：mean 聚合首先会对邻居节点进行均值聚合，然后当前节点特征与邻居节点特征该分别送入全连接网络中。
            
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
                    
            #2.gcn：先处理起点特征，再处理终点特征。 由于dstdata的neigh特征和h特征维度相同，故可直接相加，在进行归一化处理。
            
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    if graph.is_block:
                        graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))  #消息传递
                
                # 除以度
                
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
                    
            #3.pool：池化方法中，每一个节点的向量都会对应一个全连接神经网络，然后基于元素排列取最大池化操作。
            
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
                
            #4.lstm：先准备起点h特征，然后通过每个节点的mailbox数据送到lstm，将h_n结果赋给终点的neigh特征
            
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
                
   #消息聚合
    
            #如果是非gcn聚合函数，则需加上自身特征。
        
            if self._aggre_type == 'gcn':
                rst = h_neigh
            else:
                rst = self.fc_self(h_self) + h_neigh
                
   #输出更新后的特征

            # bias term
    
            if self.bias is not None:
                rst = rst + self.bias

            # 激活函数
            
            if self.activation is not None:
                rst = self.activation(rst)
                
            # 归一化
            
            if self.norm is not None:
                rst = self.norm(rst)
            return rst

#单层SAGE
class R_SAGE(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,rel_names, self_loop=False):
        super().__init__()
        
        self.conv1 = dglnn.HeteroGraphConv({ rel_names[0]: sageconv(in_feats, hid_feats,'mean'),
                                           rel_names[1]: sageconv(in_feats, hid_feats,'mean'),
                                           rel_names[2]: GCNlayer(in_feats, hid_feats)

                                           }, aggregate='sum')
        

        self.conv2 = dglnn.HeteroGraphConv({ rel_names[0]: sageconv(hid_feats, out_feats,'mean'),
                                           rel_names[1]: sageconv(hid_feats, out_feats,'mean'),
                                           rel_names[2]: GCNlayer(hid_feats, out_feats)

                                           }, aggregate='sum')
    def forward(self, graph, inputs):
        
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class LinkModel_s(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats,rel_names):
        super().__init__()
        self.rsage = R_SAGE(in_feats, hidden_feats,
                      out_feats,rel_names)
        
        self.pred = HereroCatPredictor()
        
    def forward(self, pos_g, neg_g, x, etype):
        
        # x代表的是整个图的初始特征-node_features。
        h = self.rsage(pos_g, x)
        
        return self.pred(pos_g, h, x, etype), self.pred(neg_g, h, x, etype)
# #单层模型_cat
# class LinkModel_s_cat_32(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats, rel_names):
#         super().__init__()
#         self.rsage = R_SAGE(in_feats, hid_feats,
#                       out_feats, rel_names)
#         self.sequence = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(16, 1)
#         )
#         self.predict = nn.Sigmoid()
        
#     def forward(self, g, x, nodes):
#         h = self.rsage(g, x)
#         for i in range(len(nodes)):
#             pinjie = torch.cat((h['gene'][nodes[i][0]],h['disease'][nodes[i][1]]),0)
#             if i==0:
#                 input_ = pinjie.unsqueeze(0)
#             else:
#                 input_ = torch.cat((input_, pinjie.unsqueeze(0)), 0)
   
#         linear_out = self.sequence(input_)
#         out = self.predict(linear_out)
        
#         return out 

#双层SAGE
class MP_R_SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, self_loop=False):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({rel_names[0]: sageconv(in_feats, hid_feats,'mean'),
                                            rel_names[1]: sageconv(in_feats, hid_feats,'mean'),
                                            rel_names[2]: GCNlayer(in_feats, hid_feats),
                                            rel_names[3]: GCNlayer(in_feats, hid_feats),
                                            rel_names[4]: GCNlayer(in_feats, hid_feats),
                                            rel_names[5]: GCNlayer(in_feats, hid_feats)
                                            
                                           }, aggregate='sum')
        
        self.conv2 = dglnn.HeteroGraphConv({rel_names[0]: sageconv(hid_feats, out_feats,'mean'),
                                            rel_names[1]: sageconv(hid_feats, out_feats,'mean'),
                                            rel_names[2]: GCNlayer(hid_feats, out_feats),
                                            rel_names[3]: GCNlayer(hid_feats, out_feats),
                                            rel_names[4]: GCNlayer(hid_feats, out_feats),
                                            rel_names[5]: GCNlayer(hid_feats, out_feats)
                                            
                                           }, aggregate='sum')
    def forward(self, graph, inputs):

        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        
        return h    
    
#双层_cat
class LinkModel_d_cat(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.rsage = MP_R_SAGE(in_feats, hid_feats,
                      out_feats, rel_names)
        self.sequence = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(8, 1)
        )
        #self.predict = nn.Sigmoid()
        
    def forward(self, g, x, nodes):
        h = self.rsage(g, x)
        for i in range(len(nodes)):
            pinjie = torch.cat((h['gene'][nodes[i][0]],h['disease'][nodes[i][1]]),0)
            if i==0:
                input_ = pinjie.unsqueeze(0)
            else:
                input_ = torch.cat((input_, pinjie.unsqueeze(0)), 0)
   
        linear_out = self.sequence(input_)
        #out = self.predict(linear_out)
        
        return linear_out