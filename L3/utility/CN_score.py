import sys
import argparse
from config import parse_args
import numpy as np
import pandas as pd
from tqdm import tqdm

# edges_route:'../data/gene_gene_num.csv'
# pairs_L3_route:'../data/pairs_L3.csv'
# path_L3_route:'../data/path_L3.csv'

def CN(args):
    print('-----------------Run CN program-----------------')
    edges = pd.read_csv(args.edges_route)
    A = np.zeros((15964, 15964))


    for i,row in edges.iterrows():

        A[row[0],row[1]] = 1
        A[row[1],row[0]] = 1

    A2 = np.matmul(A, A)

    indices_A2 = np.argwhere(np.triu(A2, k=1) > 0).tolist()

    neighbors_2 = {}
    for element in tqdm(indices_A2, total=len(indices_A2)):

        if element[0] not in neighbors_2:
            neighbors_2[element[0]] = [element[1]]
        else:
            neighbors_2[element[0]].append(element[1])

        if element[1] not in neighbors_2:
            neighbors_2[element[1]] = [element[0]]
        else:
            neighbors_2[element[1]].append(element[0])

    A3 = np.matmul(A,A2)

    indices = np.argwhere(np.triu(A3, k=1) > 0).tolist()


    result_L1 = {}
    for i in tqdm(range(len(A))):
        for j in range(len(A)):
            if A[i][j] != 0:
                if i not in result_L1:
                    result_L1[i] = {j}
                else:
                    result_L1[i].add(j)

    pairs_L3 = []
    neighbors_1 = result_L1
    for element in tqdm(indices):

        if element[1] in neighbors_1[element[0]]:
            continue
        if element[1] in neighbors_2[element[0]]:
            continue

        pairs_L3.append(element)

    the_df = pd.DataFrame(pairs_L3)

    score = []
    for i in tqdm(pairs_L3):
        score.append(A3[i[0]][i[1]])


    the_df['num'] = score
    the_df.columns = ['node1', 'node2', 'num']
    the_df.to_csv(args.pairs_L3_route, index = False)

    data = the_df
    data1 = data['node1'].tolist()
    data2 = data['node2'].tolist()
    pairs_L3 = list(zip(data1, data2))


    ls_sum = []
    for i in tqdm(pairs_L3):

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
    edge_.to_csv(args.path_L3_route, index= False)
    print('----------------- CN program over-----------------')


def main():
    args = parse_args()
    CN(args)

if __name__ == '__main__':
    main()