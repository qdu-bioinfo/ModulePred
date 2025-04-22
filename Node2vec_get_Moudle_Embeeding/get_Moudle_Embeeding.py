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


data_use = pd.read_csv('./Output/used_human_complexes.csv')

df = data_use.dropna(subset=['Disease'])
data1 = df.reset_index(drop=True)

lenss = len(data1['subunits(Gene name)'])


protein_cp = {}
for i in range(lenss):
    protein_cp.setdefault(i, []) 
    tmp1 = data1.iloc[i]['subunits(Gene name)']
    tmp2 = data1.iloc[i]['Disease']
    tmp = tmp1 + ';' + tmp2
    if ';' in tmp:
        tmp = tmp.split(';')
        for t in tmp:
            protein_cp[i].append(t) 
    else:
        protein_cp[i].append(tmp)


ID_pc = {}
for m,n in protein_cp.items():
    for i in n:
        ID_pc.setdefault(i,m)

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

f = open('./Input/_dict_gene.txt', 'r')
geneNumDic = eval(f.read())
f.close()
print(type(geneNumDic), len(geneNumDic))

f = open('./Input/_dict_d.txt', 'r')
disNumDic = eval(f.read())
f.close()
print(type(disNumDic), len(disNumDic))


geneList = list(geneNumDic.keys())
disList = list(disNumDic.keys())


data_gene_dis = pd.read_csv('./Input/disease_gene_edges_random_shuffled.csv')


df1 = pd.read_csv('../data/gene_gene.csv')
df2 = pd.read_csv('./Input/gene_gene_L3.csv')
data_gene = pd.concat([df1, df2], axis=0, ignore_index=True)

G = nx.Graph()
G.add_nodes_from(disList)
G.add_nodes_from(geneList)


for ggg in tqdm(range(len(data_gene))):
    g1 = data_gene.loc[ggg]['gene1']
    g2 = data_gene.loc[ggg]['gene2']
    G.add_edges_from([(g1, g2)])
for ddd in tqdm(range(len(data_gene_dis))):
    gpg = data_gene_dis.loc[ddd]['gene']
    gpd = data_gene_dis.loc[ddd]['disease']
    G.add_edges_from([(gpg, gpd)])

nodes = G.nodes

print(len(G.nodes()))
print(len(G.edges()))

print(type(nodes), len(nodes))

model = Node2Vec(G, 128, 64, 10, 0.3, 0.7)
walks_nopc = model.walks

kwargssss = {"sentences": walks_nopc, "min_count": 0, "vector_size": 128, "sg": 1, "hs": 0, "workers": 3,
                "window": 3, "epochs": 10}
model_walk_nopc = Word2Vec(**kwargssss)

bionet_embeddings = get_bionet_embeddings(model_walk_nopc, G)

disEmbeddingsDic = {}
geneEmbeddingsDic = {}

for xxx in tqdm(bionet_embeddings.keys()):
    if xxx in disNumDic.keys():
        key = disNumDic[xxx]
        value = list(bionet_embeddings[xxx])
        disEmbeddingsDic.setdefault(key, value)

    elif xxx in geneNumDic.keys():
        key = geneNumDic[xxx]
        value = list(bionet_embeddings[xxx])
        geneEmbeddingsDic.setdefault(key, value)

print('disList:', len(disList), 'disEmbeddingsDic:', len(disEmbeddingsDic))
print('geneList:', len(geneList), 'geneEmbeddingsDic:', len(geneEmbeddingsDic))


f = open('./Output/dis_EmbeddingsDic.txt', 'w')
f.write(str(disEmbeddingsDic))
f.close()
print(len(disEmbeddingsDic))

f = open('./Output/gene_EmbeddingsDic.txt', 'w')
f.write(str(geneEmbeddingsDic))
f.close()
print(len(geneEmbeddingsDic))


new_walk = []
lines = 0
for yyy in tqdm(walks_nopc):
    new_walk.append([])
    for yyds in range(len(yyy)):

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
pc_embeddings = get_pc_embeddings(model_walk_pc, embeddingList)


module_feature = {}
for ttt in bionet_embeddings.keys():
    module_feature.setdefault(ttt, [])
    if ttt in ID_pc.keys() and str(ID_pc[ttt]) in embeddingList:
        c = np.hstack((bionet_embeddings[ttt], pc_embeddings[str(ID_pc[ttt])]))

        module_feature[ttt] = c

    else:
        e = np.hstack((bionet_embeddings[ttt], pc_embeddings['-1']))
        module_feature[ttt] = e

disEmbeddingsDic_module = {}
geneEmbeddingsDic_module = {}

for fff in tqdm(module_feature.keys()):
    if fff in disNumDic.keys():
        key = disNumDic[fff]
        value = list(module_feature[fff])
        disEmbeddingsDic_module.setdefault(key, value)

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


tezheng = pd.DataFrame(feature_list, index=fea_name_list, columns=list1)

dis_feature = tezheng.head(13074)
dis_feature.to_csv('./Output/dis_feature_256.csv')
gene_feature = tezheng[13074:]
gene_feature.to_csv('./Output/gene_feature_256.csv')





f = open('./Input/_dict_gene.txt', 'r')
_dict_gene = eval(f.read())
f.close()

f = open('./Input/_dict_d.txt', 'r')
_dict_d = eval(f.read())
f.close()

feature_d = pd.read_csv('./Output/dis_feature_256.csv')
feature_d['Unnamed: 0'] = feature_d['Unnamed: 0'].map(_dict_d)

feature_d.to_csv('./Output/dis_feature_256_num.csv', index=False)

feature_g = pd.read_csv('./Output/gene_feature_256.csv')
feature_g['Unnamed: 0'] = feature_g['Unnamed: 0'].map(_dict_gene)

feature_g.to_csv('./Output/gene_feature_256_num.csv', index=False)
