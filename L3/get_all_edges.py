import pandas as pd


df = pd.read_csv('../data/gene_gene.csv')
df_L3 = pd.read_csv('./Output/gene_gene_L3.csv')
df_all = pd.concat([df,df_L3]).reset_index(drop=True)


df_d2g = pd.read_csv('../data/disease_gene_edge.csv')
node_map = pd.read_csv('../data/all_node_map.txt',sep='\t',header=None)
node2num = dict(zip(node_map[0],node_map[1]))


df_all['node1'] = df_all['gene1'].apply(lambda x: node2num[x])
df_all['node2'] = df_all['gene2'].apply(lambda x: node2num[x])

df_d2g['node1'] = df_d2g['disease'].apply(lambda x: node2num[x])
df_d2g['node2'] = df_d2g['gene'].apply(lambda x: node2num[x])

all_edges = pd.concat([df_d2g[['node1','node2']],df_all[['node1','node2']]]).reset_index(drop=True)

all_edges.to_csv('Output/all_node_edge_num.txt',header=None,sep='\t',index=False)