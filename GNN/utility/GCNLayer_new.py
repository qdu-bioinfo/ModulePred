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
        
class GCNlayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight= True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GCNlayer, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation



    def reset_parameters(self):

        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value



    def forward(self, graph, feat, weight=None, edge_weight=None):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # 为了不同的度矩阵需求，对degs和norm进行不同的处理，both情况下norm需要开平方，其他情况只需要取逆.
            
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            #输入特征维度高于输出维度，先右乘权重矩阵再进行消息传递,这样可以在传递前减少需要计算的维度.否则先消息传递，再乘权重矩阵.
            
            if self._in_feats > self._out_feats:
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)      
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


    def extra_repr(self):

        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
#------------------------R_GCN------------------------------#
#单层GCN
class R_GCN(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,rel_names, self_loop=False):
        super().__init__()
        
        self.conv1 = dglnn.HeteroGraphConv({ rel_names[0]: GCNlayer(in_feats, hid_feats),
                                           rel_names[1]: GCNlayer(in_feats, hid_feats),
                                           rel_names[2]: GCNlayer(in_feats, hid_feats)

                                           }, aggregate='sum')
        

        self.conv2 = dglnn.HeteroGraphConv({ rel_names[0]: GCNlayer(hid_feats, out_feats),
                                           rel_names[1]: GCNlayer(hid_feats, out_feats),
                                           rel_names[2]: GCNlayer(hid_feats, out_feats)

                                           }, aggregate='sum')
    def forward(self, graph, inputs):
        
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        
        return h
#------------------------点乘积-1------------------------------#  
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # graph 图对象
        # etype 边类型
        
        #对于训练之后的h--train_features，通过torch.norm得到对应图中，每一个节点的模长值
        inputs_d = torch.norm(h['disease'][0:], p = 2, dim = 1, keepdim = True)
        inputs_g = torch.norm(h['gene'][0:], p = 2, dim = 1, keepdim = True)
        
        m = {'gene': inputs_g, 'disease': inputs_d}

        with graph.local_scope():
        # 一次性为所有节点类型的 'h'赋值
            graph.ndata['h'] = h 
            # 使用 点积 计算得分--余弦相似性的分子
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

            _dot = graph.edges[etype].data['score']

            # 一次性为所有节点类型的 'h'赋值
            graph.ndata['h'] = m  
            # 使用 点积 计算得分--余弦相似性的分母
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

            _mou = graph.edges[etype].data['score']
            
            #返回结果，即为：对应的余弦相似性
            return _dot/_mou   

        
#单层点积运算
class LinkModel_s_yuan(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,rel_names):
        
        super().__init__()
        self.rgcn = R_GCN(in_feats,hid_feats,out_feats,rel_names)
        self.pred = HeteroDotProductPredictor()
    
    def forward(self, pos_g, neg_g, x, etype):
        
        # x代表的是整个图的初始特征-node_features。
        h = self.rgcn(pos_g, x)
        
        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype)
#------------------------点乘积-2------------------------------#  
#单层点积运算
class LinkModel_s_dot(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,rel_names):
        
        super().__init__()
        self.rgcn = R_GCN(in_feats,hid_feats,out_feats,rel_names)
        self.pred = HereroCosPredictor()
    
    def forward(self, pos_g, neg_g, x, etype):
        
        # x代表的是整个图的初始特征-node_features。
        h = self.rgcn(pos_g, x)
        
        return self.pred(pos_g, h, x, etype), self.pred(neg_g, h, x, etype)
    
    
#------------------------原来的特征和新的h进行拼接------------------------------#
# 将原来的特征和新的h进行拼接
class HereroCatPredictor(nn.Module):
    def forward(self, graph, h, x, etype):
        # graph 图对象
        # h是训练完毕的特征
        # m是得到的每个节点的模长--也是一个处理完毕之后的字典
        # etype 边类型
        
        # _features_new代表的是初始特征与最终的训练特征进行点乘得到的
        _features_new = {}
        for key in x.keys():
            
            hs = torch.cat([x[key], h[key]], dim=1)
            
            _features_new[key] = hs
        
        #对于训练之后的h--train_features，通过torch.norm得到对应图中，每一个节点的模长值
        inputs_d = torch.norm(_features_new['disease'][0:], p = 2, dim = 1, keepdim = True)
        inputs_g = torch.norm(_features_new['gene'][0:], p = 2, dim = 1, keepdim = True)
        
        m = {'gene': inputs_g, 'disease': inputs_d}
        
        with graph.local_scope():
        # 一次性为所有节点类型的 'h'赋值
            graph.ndata['h'] = _features_new 
            # 使用 点积 计算得分--余弦相似性的分子
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

            _dot = graph.edges[etype].data['score']

            # 一次性为所有节点类型的 'h'赋值
            graph.ndata['h'] = m  
            # 使用 点积 计算得分--余弦相似性的分母
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

            _mou = graph.edges[etype].data['score']
            
            #返回结果，即为：对应的余弦相似性
            return _dot/_mou   

#单层原特征与GNN特征拼接运算
class LinkModel_s(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,rel_names):
        
        super().__init__()
        self.rgcn = R_GCN(in_feats,hid_feats,out_feats,rel_names)
        self.pred = HereroCatPredictor()
    
    def forward(self, pos_g, neg_g, x, etype):
        
        # x代表的是整个图的初始特征
        h = self.rgcn(pos_g, x)
        
        return self.pred(pos_g, h, x, etype), self.pred(neg_g, h, x, etype)
    

    
#------------------------原来的特征和新的h进行拼接-New------------------------------#
# 将原来的特征和新的h进行拼接
class HereroCatPredictor_new(nn.Module):
    def forward(self, graph, h, LVR_f, etype):
        # graph 图对象
        # h是训练完毕的特征
        # m是得到的每个节点的模长--也是一个处理完毕之后的字典
        # etype 边类型
        
        # _features_new代表的是初始特征与最终的训练特征进行点乘得到的
        _features_new = {}
        for key in LVR_f.keys():
            
            hs = torch.cat([LVR_f[key], h[key]], dim=1)
            
            _features_new[key] = hs
        
        #对于训练之后的h--train_features，通过torch.norm得到对应图中，每一个节点的模长值
        inputs_d = torch.norm(_features_new['disease'][0:], p = 2, dim = 1, keepdim = True)
        inputs_g = torch.norm(_features_new['gene'][0:], p = 2, dim = 1, keepdim = True)
        
        m = {'gene': inputs_g, 'disease': inputs_d}
        
        with graph.local_scope():
        # 一次性为所有节点类型的 'h'赋值
            graph.ndata['h'] = _features_new 
            # 使用 点积 计算得分--余弦相似性的分子
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

            _dot = graph.edges[etype].data['score']

            # 一次性为所有节点类型的 'h'赋值
            graph.ndata['h'] = m  
            # 使用 点积 计算得分--余弦相似性的分母
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

            _mou = graph.edges[etype].data['score']
            
            #返回结果，即为：对应的余弦相似性
            return _dot/_mou        
#单层原特征与GNN特征拼接运算
class LinkModel_s_new(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,rel_names):
        
        super().__init__()
        self.rgcn = R_GCN(in_feats,hid_feats,out_feats,rel_names)
        self.pred = HereroCatPredictor_new()
    
    def forward(self, pos_g, neg_g, x, LVR_f, etype):
        
        # x代表的是整个图的初始特征。LVR_f代表的是LVR的特征
        h = self.rgcn(pos_g, x)
        
        return self.pred(pos_g, h, LVR_f, etype), self.pred(neg_g, h, LVR_f, etype)

        