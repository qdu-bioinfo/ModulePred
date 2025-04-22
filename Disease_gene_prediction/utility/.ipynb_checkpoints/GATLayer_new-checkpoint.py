import torch.nn as nn
import torch.nn.functional as F
import torch
import torch as th
from torch import nn
from torch.nn import init

import dgl
import dgl.nn as dglnn
import dgl.function as fn

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.transform import reverse
from dgl.convert import block_to_graph
from dgl.heterograph import DGLBlock

from utility.GCNLayer_new import GCNlayer
from utility.GCNLayer_new import HereroCatPredictor_new

import math

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

#         self.attn_dropout = nn.Dropout(0.1)#(attention_probs_dropout_pro)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)         #218*7*7  1052*7*7

        context_layer = torch.matmul(attention_probs, value_layer)
                     #218*7*3  1052*7*3
        context_layer = context_layer.permute(0, 2, 1).contiguous()
           #218*3*7  218*3*7
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)

        return hidden_states
#单层
class R_GAT(nn.Module):
    def __init__(self, in_feats,infeats_1, hidden_feats, out_features, rel_names):
        
        super().__init__()
        
        self.layers = nn.ModuleList()

        self.layers.append(SelfAttention(2,in_feats,infeats_1,hidden_dropout_prob=0))
        self.layers.append(dglnn.HeteroGraphConv({rel_names[0]: GCNlayer(infeats_1, hidden_feats),
                                            rel_names[1]: GCNlayer(infeats_1, hidden_feats),
                                                 rel_names[2]: GCNlayer(infeats_1, hidden_feats)}, 
                                                 aggregate='sum')) 
        self.layers.append(dglnn.HeteroGraphConv({rel_names[0]: GCNlayer(hidden_feats, out_features),
                                            rel_names[1]: GCNlayer(hidden_feats, out_features),
                                                  rel_names[2]: GCNlayer(hidden_feats, out_features)
                                           }, aggregate='sum'))
    
    def forward(self, g,  h):

        h1=self.layers[0](h['gene'])
        h2=self.layers[0](h['disease'])
        h={'gene':h1,'disease':h2}
        h = {k: F.relu(v) for k, v in h.items()}
        h=self.layers[1](g,h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.layers[2](g,h)
        return h


#单层模型
class LinkModel_s_new(nn.Module):
    def __init__(self, in_feats,infeats_1, hidden_feats, out_feats,rel_names):
        super().__init__()
        self.rgat = R_GAT(in_feats, infeats_1,hidden_feats,
                      out_feats,rel_names)
        
        self.pred = HereroCatPredictor_new()
        
    def forward(self, pos_g, neg_g, x, LVR_f, etype):
        
        # x代表的是整个图的初始特征-node_features。
        h = self.rgat(pos_g, x)
        
        return self.pred(pos_g, h, LVR_f, etype), self.pred(neg_g, h, LVR_f, etype)

