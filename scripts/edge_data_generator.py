import sys

# relative path
sys.path.append('../')

from utils.multiplex import Multiplex
from utils.textualizer import textualize_edges
from utils.array_sampler import array_sampler
import pandas as pd
import numpy as np
import os.path

def edge_data(multiplex_file: str,
              output_file: str,
              num_tests: int = -1,
              balance_tests: bool = True):
  # Load multiplex
  mp = Multiplex(multiplex_file)
  N = mp.num_nodes
  L = len(mp)

  # Calculate maximum number of pairs for nodes (i,j) where i != j
  max_pairs = N * (N - 1)
  # Account for asking per layer
  max_pairs *= (L)

  # Limit the number of tests
  if num_tests == -1:
    print(f'Setting number of edge tests to {max_pairs}')
    num_tests = max_pairs
  if num_tests > max_pairs:
    print(f'Limiting number of edge tests to {max_pairs} to avoid duplication.')
    num_tests = max_pairs

  # Calculate number of possible positive tests per layer
  n_pos_tests = 0
  for layer in mp.layers:
    n_pos_tests += (2 * layer['graph'].number_of_edges())
  n_neg_tests = max_pairs - n_pos_tests
  
  # Check if balancing positive and negative tests
  if balance_tests:
    print('Balancing positive and negative tests')
    # Check is tests can be balanced while meeting 'num_tests' requirement
    min_pos_neg = min([n_pos_tests, n_neg_tests])

    if 2 * min_pos_neg > num_tests:
      print(f'Cannot balance positive and negative tests while generating {num_tests} tests.')
      
      min_pos_neg = int(num_tests/2)
      print(f'Reducing number of tests to {2*min_pos_neg}')

    n_pos_tests = min_pos_neg
    n_neg_tests = min_pos_neg
  else:
    # Reduce the number of positive and negative tests while keeping same ratio
    n_pos_tests = int(n_pos_tests / max_pairs * num_tests)
    n_neg_tests = num_tests - n_pos_tests

  print(f'Creating {n_pos_tests} positive tests and {n_neg_tests} negative tests')

  tests = pd.DataFrame(None, columns=['type','pos_test','question','label','desc'])

  # Create edge list
  edge_list = textualize_edges(mp)
  # Write edge list to file
  edge_list_file = os.path.splitext(multiplex_file)[0] + '.mpedgelist'
  with open(edge_list_file, "w") as fp:
    fp.write(edge_list)

  # Add positive tests
  for layer in mp.layers:
    edges = list(layer['graph'].edges())
    edges_idx = np.array([i for i in range(2*len(edges))])
    pos_sampler = array_sampler(edges_idx)
    n_tests = 0
    while n_tests < n_pos_tests:
      idx = pos_sampler.sample()
      if mp[0]['graph'].is_directed():
        u, v = edges[idx]
      else:
        if idx >= len(edges):
          idx = idx % len(edges)
          v, u = edges[idx]
        else:
          u, v = edges[idx]

      question = f"Is there an edge between nodes {u} and {v} in layer {layer['layer_name']}?"
      label = ['yes']
      tests.loc[len(tests)] = ['edge', 1, question, label, edge_list_file]
      n_tests += 1
  
  # Add negative tests
  pair_idxs = np.array([i for i in range(N*N)])
  nodes = mp.nodes
  n_tests = 0
  for layer in mp.layers:
    sampler = array_sampler(pair_idxs)

    while n_tests < n_neg_tests:
      pair_idx = sampler.sample()

      u_idx = int(pair_idx / N)
      v_idx = pair_idx % N
      u = nodes[u_idx]
      v = nodes[v_idx]

      if u == v:
        pass
      elif layer['graph'].has_edge(u, v):
        pass
      else:
      
        question = f"Is there an edge between nodes {u} and {v} in layer {layer['layer_name']}?"
        label = ['no']
        tests.loc[len(tests)] = ['edge', 0, question, label, edge_list_file]
        n_tests += 1

  tests = tests.sample(frac=1)

  # Write tests to file
  add_headers = not os.path.exists(output_file)
  tests.to_csv(output_file, mode='a', sep='\t', index=False, header=add_headers)


if __name__ == '__main__':
  multiplex_file = '../bio_tissue_networks.flist'
  output_file = 'edge_tests.tsv'

  edge_data(multiplex_file, output_file)
