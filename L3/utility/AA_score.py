import numpy as np
import pandas as pd
from tqdm import tqdm
from config import parse_args

def AA(args):
    print('----------------- Run AA program-----------------')
    edges = pd.read_csv(args.edges_route)

    A = np.zeros((29, 29))
    for i,row in edges.iterrows():

        A[row[0],row[1]] = 1
        A[row[1],row[0]] = 1

    data = pd.read_csv(args.path_L3_route)
    path = data['path'].tolist()

    #   计算得分
    aa_list = []
    for nei in tqdm(path):
        aa_sim = 0
        for index, node in enumerate(eval(nei)):
            if A[node].sum() != 1:
                aa_sim += 1 / np.log(A[node].sum())
        aa_list.append(aa_sim)

    data['AA'] = aa_list
    data.to_csv(args.path_L3_AA, index = False)
    print('-----------------AA program over-----------------')
def main():
    args = parse_args()
    AA(args)

if __name__ == '__main__':
    main()