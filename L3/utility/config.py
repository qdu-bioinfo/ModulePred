import argparse

# edges_route:'../../data/L3/gene_gene_num.csv'
# pairs_L3_route:'../../data/L3/pairs_L3.csv'
# path_L3_route:'../../data/L3/path_L3.csv'
# path_L3_RA:'../../data/L3/path_L3_RA.csv'
# path_L3_AA:'../../data/L3/path_L3_AA.csv'
# candidate_edges:'../../data/L3/L3_edges.txt'
def parse_args():
    parser = argparse.ArgumentParser(description='virutal-edges')

    # Data settings
    parser.add_argument('--edges_route', type=str, default='./Output/gene_gene_num.csv')
    parser.add_argument('--pairs_L3_route', type=str, default='./Output/pairs_L3.csv')
    parser.add_argument('--path_L3_route', type=str, default='./Output/path_L3.csv')
    parser.add_argument('--path_L3_RA', type=str, default='./Output/path_L3_RA.csv')
    parser.add_argument('--path_L3_AA', type=str, default='../Output/path_L3_AA.csv')
    parser.add_argument('--candidate_edges', type=str, default='./Output/gene_gene_L3.csv')
    parser.add_argument('--gene_num_return_dict', type=str, default='./Output/gene_num_return_dict.txt')

    return parser.parse_args()