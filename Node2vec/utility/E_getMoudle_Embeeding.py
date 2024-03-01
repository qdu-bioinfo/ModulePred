from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import csv
import os
import datetime
import time
from node2vec import Node2Vec
from gensim.models import Word2Vec


#-----------------------------------------------------------------------------------------------------------------------
# 获取复合物与其对应的复合物ID
data_use = pd.read_csv('./data/Node2vec/used_human_complexes.csv')
#   删除'Disease'列中缺失值（NaN）的行,并重置索引
df = data_use.dropna(subset=['Disease'])
data1 = df.reset_index(drop=True)

lenss = len(data1['subunits(Gene name)'])

#   初始化一个空字典以存储蛋白质复合物
protein_cp = {}
for i in range(lenss):
    protein_cp.setdefault(i, [])    #   为每个蛋白质复合物初始化一个空列表
    tmp1 = data1.iloc[i]['subunits(Gene name)']
    tmp2 = data1.iloc[i]['Disease']
    tmp = tmp1 + ';' + tmp2
    if ';' in tmp:
        tmp = tmp.split(';')
        for t in tmp:
            protein_cp[i].append(t) #   将每个疾病和基因添加到蛋白质复合物字典中
    else:
        protein_cp[i].append(tmp)   #   如果只有一个基因或疾病，将其添加到蛋白质复合物字典中

#   ID_pc键是蛋白质和疾病，值是它所出现的复合物，同一个键可能会有多个不同的值，对应着同一个蛋白质或者疾病，可能在不同的复合物中出现
ID_pc = {}
for m,n in protein_cp.items():
    for i in n:
        ID_pc.setdefault(i,m)

#-----------------------------------------------------------------------------------------------------------------------
#   获取生物分子网络节点embedding
def get_bionet_embeddings(w2v_model, graph):
    count = 0
    invalid_word = []
    _embeddings = {}
    for word in graph.nodes():
        if word in w2v_model.wv:
            _embeddings[word] = w2v_model.wv[word]
        else:
            invalid_word.append(word)
            count += 1

    return _embeddings


#   获取复合物节点embedding
def get_pc_embeddings(w2v_model, embeddingList):
    count = 0
    invalid_word = []
    _embeddings = {}
    for word in embeddingList:
        if word in w2v_model.wv:
            _embeddings[word] = w2v_model.wv[word]
        else:
            invalid_word.append(word)
            count += 1


    return _embeddings

#-----------------------------------------------------------------------------------------------------------------------
# 将数据集划分为：蛋白质和其对应的num 基因和其对应的num  存储是以便于后续图神经网络建图 图神经网络中必须要以连续数字建图

#   加载基因编号字典 geneNumDic
f = open('./data/Node2vec/_dict_gene.txt', 'r')
geneNumDic = eval(f.read())
f.close()
print(type(geneNumDic), len(geneNumDic))

#   加载疾病编号字典 disNumDic
f = open('./data/Node2vec/_dict_d.txt', 'r')
disNumDic = eval(f.read())
f.close()
print(type(disNumDic), len(disNumDic))

#   加载疾病节点、基因节点
geneList = list(geneNumDic.keys())
disList = list(disNumDic.keys())

#   获得训练集的基因-疾病连边 data_gene_dis
dg = pd.read_csv('./data/Node2vec/new_cv.txt')
dg = dg[dg['index'] == 'train']
data_gene_dis = dg.drop(columns='index')
data_gene_dis = data_gene_dis.reset_index(drop=True)
print("基因疾病网络：", data_gene_dis.shape)

#   基因网络 data_gene （真实连边 + 虚拟连边）
data_gene = pd.read_csv('./data/Node2vec/all_gene_node_edges.csv')  # 基因网络不变
# data_gene = pd.read_csv('use_DGN/gene_gene.csv')   #   基因网络
print("基因网络：", data_gene.shape)

#   创建关系网络（基因-基因，疾病-基因）
G = nx.Graph()
G.add_nodes_from(disList)  # 添加疾病节点
G.add_nodes_from(geneList)  # 添加基因节点

#   添加 基因-基因 连边
for ggg in tqdm(range(len(data_gene))):
    g1 = data_gene.loc[ggg]['gene1']
    g2 = data_gene.loc[ggg]['gene2']
    G.add_edges_from([(g1, g2)])
#   添加 疾病-基因连边
for ddd in tqdm(range(len(data_gene_dis))):
    gpg = data_gene_dis.loc[ddd]['gene']
    gpd = data_gene_dis.loc[ddd]['disease']
    G.add_edges_from([(gpg, gpd)])

nodes = G.nodes  # 获取图G的顶点

#   关系矩阵 R 的行列
print(type(nodes), len(nodes))

#   定义node2vec模型--时间较长
model = Node2Vec(G, 128, 64, 10, 0.3, 0.7)
walks_nopc = model.walks

kwargssss = {"sentences": walks_nopc, "min_count": 0, "vector_size": 128, "sg": 1, "hs": 0, "workers": 3,
                "window": 3, "epochs": 10}
model_walk_nopc = Word2Vec(**kwargssss)

bionet_embeddings = get_bionet_embeddings(model_walk_nopc, G)

#   疾病节点Embeeding
disEmbeddingsDic = {}
#   基因节点Embeeding
geneEmbeddingsDic = {}

for xxx in tqdm(bionet_embeddings.keys()):
    # dis
    if xxx in disNumDic.keys():
        key = disNumDic[xxx]
        value = list(bionet_embeddings[xxx])
        disEmbeddingsDic.setdefault(key, value)

    # 基因
    elif xxx in geneNumDic.keys():
        key = geneNumDic[xxx]
        value = list(bionet_embeddings[xxx])
        geneEmbeddingsDic.setdefault(key, value)

print('disList:', len(disList), 'disEmbeddingsDic:', len(disEmbeddingsDic))
print('geneList:', len(geneList), 'geneEmbeddingsDic:', len(geneEmbeddingsDic))

#   存储疾病和基因节点embedding文件
f = open('./data/Node2vec/dis_EmbeddingsDic.txt', 'w')
f.write(str(disEmbeddingsDic))
f.close()
print(len(disEmbeddingsDic))

f = open('./data/Node2vec/gene_EmbeddingsDic.txt', 'w')
f.write(str(geneEmbeddingsDic))
f.close()
print(len(geneEmbeddingsDic))

#   基因游走序列替换对应的模体ID
#   这段代码并不是完善，检测到基因或者疾病在某个模体中后，会将游走序列中的基因或疾病替换为模体编号，但是如果基因或疾病存在于多个模体之中时，会将其随机替换为多个模体中的某个
new_walk = []
lines = 0
for yyy in tqdm(walks_nopc):
    new_walk.append([])
    for yyds in range(64):

        if yyy[yyds] in ID_pc.keys():
            new_walk[lines].append(str(ID_pc[yyy[yyds]]))

        else:
            new_walk[lines].append(str(-1))
    lines = lines + 1

kwarg = {"sentences": new_walk, "min_count": 0, "vector_size": 128, "sg": 1, "hs": 0, "workers": 2, "window": 5,
            "epochs": 10}
model_walk_pc = Word2Vec(**kwarg)

pc_list = []
for asd in range(8):
    pc_list.append(str(asd - 1))
embeddingList = pc_list
#   模体的Embeeding -> pc_embeddings
pc_embeddings = get_pc_embeddings(model_walk_pc, embeddingList)

#   特征拼接
#   对于每个基因、疾病，如果它属于某个模体，就将其特征与模体特征进行拼接。否则以 -1 代替
#   拼接后的特征 module_feature
module_feature = {}
for ttt in bionet_embeddings.keys():
    module_feature.setdefault(ttt, [])
    if ttt in ID_pc.keys() and str(ID_pc[ttt]) in embeddingList:
        c = np.hstack((bionet_embeddings[ttt], pc_embeddings[str(ID_pc[ttt])]))

        module_feature[ttt] = c

    else:
        e = np.hstack((bionet_embeddings[ttt], pc_embeddings['-1']))
        module_feature[ttt] = e

#   拼接后的疾病特征
disEmbeddingsDic_module = {}
#   拼接后的基因特征
geneEmbeddingsDic_module = {}

for fff in tqdm(module_feature.keys()):
    # 疾病
    if fff in disNumDic.keys():
        key = disNumDic[fff]
        value = list(module_feature[fff])
        disEmbeddingsDic_module.setdefault(key, value)

    # 基因
    elif fff in geneNumDic.keys():
        key = geneNumDic[fff]
        value = list(module_feature[fff])
        geneEmbeddingsDic_module.setdefault(key, value)

print('disList:', len(disList), 'disEmbeddingsDic:', len(disEmbeddingsDic_module))

print('geneList:', len(geneList), 'geneEmbeddingsDic:', len(geneEmbeddingsDic_module))

fea_name_list = list(module_feature.keys())
feature_list = []
liness = 0
for uuu in fea_name_list:
    feature_list.append([])
    for uus in module_feature[uuu]:
        feature_list[liness].append(uus)
    liness = liness + 1

list1 = []
for mmm in range(256):
    list1.append(mmm)

# 存储基因和疾病的特征
tezheng = pd.DataFrame(feature_list, index=fea_name_list, columns=list1)

dis_feature = tezheng.head(1)
dis_feature.to_csv('./data/Node2vec/dis_feature_256.csv')
gene_feature = tezheng[1:]
gene_feature.to_csv('./data/Node2vec/gene_feature_256.csv')



# 对基因-疾病的256维特征，将索引替换为相应的编号


#   基因编号
f = open('./data/Node2vec/_dict_gene.txt', 'r')
_dict_gene = eval(f.read())
f.close()
#   疾病编号
f = open('./data/Node2vec/_dict_d.txt', 'r')
_dict_d = eval(f.read())
f.close()

feature_d = pd.read_csv('./data/Node2vec/dis_feature_256.csv')
feature_d['Unnamed: 0'] = feature_d['Unnamed: 0'].map(_dict_d)

feature_d.to_csv('./data/Node2vec/dis_feature_256_num.csv', index=False)  # 疾病特征对应编号

feature_g = pd.read_csv('./data/Node2vec/gene_feature_256.csv')
feature_g['Unnamed: 0'] = feature_g['Unnamed: 0'].map(_dict_gene)

feature_g.to_csv('./data/Node2vec/gene_feature_256_num.csv', index=False)  # 基因特征对应编号

print('=======256维特征映射新编号处理结束！=========')