# import libraries
import argparse
import os
import random
import networkx as nx

# parse arguments
def parse_distance_qa_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Network_Generator")

  parser.add_argument('--num_networks',
                      type=int,
                      required=True,
                      help='Specify the number of networks to create.')
  parser.add_argument('--num_nodes',
                      nargs='+',
                      type=int,
                      required=True,
                      help='Specify the number of nodes each network should contain. If a single value is provided, all networks have the same number of nodes.')
  parser.add_argument('--num_edges',
                      nargs='+',
                      type=int,
                      required=True,
                      help='Specify the number of edges each network should contain. If a single value is provided, all networks have the same number of edges.')
  parser.add_argument('--output_dir',
                      type=str,
                      default='./',
                      help='Specify the directory in which to store the output file(s). If no value is provided the local directory will be used.')
  parser.add_argument('--prefix',
                      type=str,
                      default='test_network',
                      help='Specify the test prefix to use in the file name. If no value is provided a default prefix of "test_network" is used.')

  return parser.parse_args()

# validate arguments
def validate_distance_qa_args(args: argparse.Namespace) -> None:
  """
  Validates arguments parsed by parse_distance_qa_args().
  Ensures:
    1) num_networks is positive
    2) num_nodes has length 1 or num_networks
    3) num_edges has length 1 or num_networks
  """
  # 1. num_networks must be positive
  if args.num_networks <= 0:
    raise ValueError(f"'num_networks' must be positive, got {args.num_networks}")

  # 2. num_nodes length must be 1 or num_networks
  if len(args.num_nodes) not in (1, args.num_networks):
    raise ValueError(
      f"'num_nodes' must have length 1 or {args.num_networks}, got {len(args.num_nodes)}"
    )

  # 3. num_edges length must be 1 or num_networks
  if len(args.num_edges) not in (1, args.num_networks):
    raise ValueError(
      f"'num_edges' must have length 1 or {args.num_networks}, got {len(args.num_edges)}"
    )

  # If all checks pass, return silently
  return

def generate_random_network(num_nodes: int, num_edges: int) -> nx.Graph:
  """
  Generate a random undirected weighted network.
  - No self-loops.
  - Random weights between 0.0 and 1.0.
  """
  G = nx.Graph()
  nodes = [f"node{i}" for i in range(1, num_nodes + 1)]
  G.add_nodes_from(nodes)

  # All possible undirected edges (u < v)
  possible_edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]

  if num_edges > len(possible_edges):
    raise ValueError(f"Too many edges requested ({num_edges}) for {num_nodes} nodes.")

  # Randomly sample unique edges
  edges = random.sample(possible_edges, num_edges)

  # Add edges with random weights
  for u, v in edges:
    G.add_edge(u, v, weight=round(random.random(), 3))

  return G


def save_network_to_file(G: nx.Graph, filepath: str):
  """Save network to a .txt file with <node1>\t<node2>\t<weight> per line."""
  with open(filepath, 'w') as f:
    for u, v, data in G.edges(data=True):
      f.write(f"{u}\t{v}\t{data['weight']:.3f}\n")


def main():
  args = parse_distance_qa_args()
  validate_distance_qa_args(args)

  os.makedirs(args.output_dir, exist_ok=True)

  for i in range(args.num_networks):
    num_nodes = args.num_nodes[0] if len(args.num_nodes) == 1 else args.num_nodes[i]
    num_edges = args.num_edges[0] if len(args.num_edges) == 1 else args.num_edges[i]

    print(f"Generating network {i+1}/{args.num_networks}: {num_nodes} nodes, {num_edges} edges")

    G = generate_random_network(num_nodes, num_edges)

    filename = f"{args.prefix}_{i+1}.txt"
    filepath = os.path.join(args.output_dir, filename)
    save_network_to_file(G, filepath)

    print(f"Saved: {filepath}")

    print("All networks generated successfully!")


if __name__ == "__main__":
    main()