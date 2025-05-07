import sys

# relative path
sys.path.append('../')

from utils.clusters import Clusters
from utils.dissimilarity_matrix import DissimilarityMatrix
from utils.multiplex import Multiplex
from utils.textualizer import textualize_edges
import pandas as pd
import os.path

def generate_gene_pair_data(clusters_file: str, dissimilarity_file: str, multiplex_file: str, output_file: str, balance_tests: bool = True):
  cluster_data = Clusters(clusters_file)
  distance_data = DissimilarityMatrix(dissimilarity_file)
  mp = Multiplex(multiplex_file)

  # Find the smallest maximum distance from each node. 
  min_max = 1.0
  for node in distance_data.nodes:
    max_dist = distance_data.get_largest_dist(node)
    if max_dist < min_max:
      min_max = max_dist
  
  # Get all distances above min_max
  neg_tests = distance_data.get_all_distance_gte(min_max)
  neg_tests['pos_test'] = False


  pos_tests = pd.DataFrame(None, columns=['node1', 'node2', 'dist'])
  # Loop through all clades
  for c in cluster_data.get_cluster_nums():
    nodes = cluster_data.get_nodes_in_cluster(c)
    pairs = distance_data.get_all_pairwise_distances(nodes)
    
    if len(pos_tests) == 0:
      pos_tests = pairs
    else:
      pos_tests = pd.concat([pos_tests, pairs])
  pos_tests['pos_test'] = True

  # Balance the number of tests
  if balance_tests:
    if len(pos_tests) > len(neg_tests):
      pos_tests = pos_tests.sample(n=len(neg_tests))
    elif len(pos_tests) < len(neg_tests):
      neg_tests = neg_tests.sample(n=len(pos_tests))
  
  tests = pd.concat([neg_tests, pos_tests])
  tests = tests.sample(frac=1)

  # Create edge list
  edge_list = textualize_edges(mp)
  # Write edge list to file
  edge_list_file = os.path.splitext(multiplex_file)[0] + '.mpedgelist'
  with open(edge_list_file, "w") as fp:
    fp.write(edge_list)
  
  # Add path to mpedgelist to df
  tests['graph'] = edge_list_file

  # Write tests to file
  tests.to_csv(output_file, mode='a', sep='\t', index=False)


if __name__ == '__main__':
  clusters_file = 'tests/data/clusters.tsv'
  dissimilarity_file = 'tests/data/dissimilarity-matrix.tsv'
  multiplex_file = 'tests/data/multiplex.flist'
  output_file = 'pair_wise_tests.tsv'

  generate_gene_pair_data(clusters_file, dissimilarity_file, multiplex_file, output_file)