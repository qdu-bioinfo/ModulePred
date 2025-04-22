import pandas as pd
from config import parse_args

def candidate(args):
    print('-----------------Run the Select Edge Program-----------------')
    data_CN = pd.read_csv(args.pairs_L3_route)
    data_RA = pd.read_csv(args.path_L3_RA)
    data_AA = pd.read_csv(args.path_L3_AA)

    def f2(x, dict):
        return dict[str(int(x))]

    node_1_sum = list(set(data_AA['node1'].tolist()))


    data_AA['rank_g'] = data_AA.groupby(['node1'])['AA'].rank(ascending = False, method = 'first')
    new_AA = data_AA[data_AA['rank_g'] <= 10]

    data_RA['rank_g'] = data_RA.groupby(['node1'])['RA'].rank(ascending = False, method = 'first')
    new_RA = data_RA[data_RA['rank_g'] <= 10]

    data_CN['rank_g'] = data_CN.groupby(['node1'])['num'].rank(ascending = False, method = 'first')
    new_CN = data_CN[data_CN['rank_g'] <= 10]


    CN_1 = new_CN['node1'].tolist()
    CN_2 = new_CN['node2'].tolist()
    node_CN = list(zip(CN_1, CN_2))

    AA_1 = new_AA['node1'].tolist()
    AA_2 = new_AA['node2'].tolist()
    node_AA = list(zip(AA_1, AA_2))

    RA_1 = new_RA['node1'].tolist()
    RA_2 = new_RA['node2'].tolist()
    node_RA = list(zip(RA_1, RA_2))

    all_edges = list(set(node_CN + node_AA + node_RA))
    len(all_edges)

    _csv = pd.DataFrame(all_edges)
    _csv.columns = ['gene1', 'gene2']

    def load_reverse_dict(reverse_dict_file):
        id_to_gene = {}
        with open(reverse_dict_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        gene_id = int(parts[0])
                        gene_name = ' '.join(parts[1:])
                        id_to_gene[gene_id] = gene_name
        return id_to_gene


    gene_dict_return = load_reverse_dict("./Output/gene_num_return_dict.txt")


    _csv['gene1'] = _csv.apply(lambda x: f2(x['gene1'], gene_dict_return), axis=1)
    _csv['gene2'] = _csv.apply(lambda x: f2(x['gene2'], gene_dict_return), axis=1)
    _csv.to_csv(args.candidate_edges, index = False)
    print('-----------------the Select Edge Program over-----------------')
def main():
    args = parse_args()
    candidate(args)

if __name__ == '__main__':
    main()