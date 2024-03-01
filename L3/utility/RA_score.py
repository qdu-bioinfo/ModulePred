from config import parse_args
import numpy as np
import pandas as pd
from tqdm import tqdm


# path_L3_RA:'../data/path_L3_RA.csv'

def RA(args):
    print('----------------- Run RA program-----------------')
    edges = pd.read_csv(args.edges_route)

    #   构造邻接矩阵
    A = np.zeros((29, 29))
    for i,row in edges.iterrows():

        A[row[0],row[1]] = 1
        A[row[1],row[0]] = 1

    #   创建1阶邻居字典
    result_L1 = {}
    for i in tqdm(range(len(A))):
        for j in range(len(A)):
            if A[i][j] != 0:
                if i not in result_L1:
                    result_L1[i] = {j}
                else:
                    result_L1[i].add(j)

    data = pd.read_csv(args.pairs_L3_route)
    data1 = data['node1'].tolist()
    data2 = data['node2'].tolist()
    pairs_L3 = list(zip(data1, data2))  # pairs_L3是距离为三的节点组合


    #   ls_sum代表每个每个距离最小为3的节点对中，节点对通路中包含的所有节点
    ls_sum = []
    for i in tqdm(pairs_L3):  # 对应第一列即可
        s1 = result_L1[i[0]] | result_L1[i[1]]
        s3 = set()
        for j in result_L1[i[0]]:
            s2 = result_L1[j] & s1
            s2.add(j)
            if len(s2) != 1:
                s3.update(s2)

        ls_sum.append(s3)

    edge_ = data.drop(columns = ['num'])
    edge_['path'] = ls_sum

    data = pd.read_csv(args.path_L3_route)
    path = data['path'].tolist()

    ra_list = []
    #   计算得分
    for nei in tqdm(path):
        ra_sim = 0
        for index, node in enumerate(eval(nei)):
            ra_sim += 1 / A[node].sum()
        ra_list.append(ra_sim)


    data['RA'] = ra_list
    data.to_csv(args.path_L3_RA, index = False)
    print('----------------- RA program over-----------------')

def main():
    args = parse_args()
    RA(args)

if __name__ == '__main__':
    main()