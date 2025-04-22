# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
import argparse
import networkx as nx
import node2vec_self
from datetime import datetime

def parse_args():

	# Parses the node2vec arguments.
	parser = argparse.ArgumentParser(description="Run node2vec.")
	parser.add_argument('--Input', nargs='?', default='graph/karate.edgelist', help='Input graph path')
	parser.add_argument('--output', nargs='?', default='walks', help='walks file name')
	parser.add_argument('--walk-length', type=int, default=10, help='Length of walk per source. Default is 10.')
	parser.add_argument('--num-walks', type=int, default=50, help='Number of walks per source. Default is 40.')
	parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
	parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is directed.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=True)
	return parser.parse_args()

def read_graph():

	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),))
	else:
		G = nx.read_edgelist(args.input, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def run_walk(args):
	# Pipeline for representational learning for all nodes in a graph.
	start_time = datetime.now()
	print('----Input parameters----')
	print('p=', args.p)
	print('q=', args.q)
	print('num_walks=', args.num_walks)
	print('walk_length=', args.walk_length)
	print('directed=', args.directed)
	print('weighted=', args.weighted)
	print('----run program----')
	nx_G = read_graph()
	print('read graph')
	G = node2vec_self.Graph(nx_G, args.directed, args.p, args.q)
	print('defined G')
	G.preprocess_transition_probs()
	print('preprocessed')
	G.simulate_walks(args.num_walks, args.walk_length, args.output, args.p, args.q)
	print('defined walk')
	end_time = datetime.now()
	print('run time: ', (end_time - start_time))


def walk_main(args):

	args.input = './Input/all_node_edge_num.txt'
	args.output = './Output/walks.txt'
	args.num_walks = 50
	args.walk_length = 50
	args.directed = False
	args.weighted = False
	args.p = 1
	args.q = 1
	print(datetime.now())
	run_walk(args)
	print(datetime.now())


if __name__ == '__main__':
	args = parse_args()
	walk_main(args)
