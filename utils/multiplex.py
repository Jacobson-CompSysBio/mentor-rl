import networkx as nx
import pandas as pd

class Multiplex:
  """Class for multiplex"""

  def __init__(self, flist: str = None):
    self.layers = []
    self._nodes = []

    if flist is not None:
      layer_info = pd.read_csv(flist, sep='\t', header=None)

      if layer_info.shape[1] < 2:
        raise ValueError('flist file must contain at least two tab seperated columns.')

      for i in layer_info.index:
        g = nx.read_edgelist(layer_info.iloc[i,0],
                              create_using=nx.Graph,
                              nodetype=str,
                              data=(('weight', float),))
        self.add_layer(g, layer_info.iloc[i,1])
  
  def __len__(self):
    """Return the number of layers in the multiplex"""
    return len(self.layers)
    
  def __getitem__(self, index):
    return self.layers[index]

  def add_layer(self, g: nx.Graph, layer_name: str):
    """Add a layer to the multiplex"""
    self.layers.append({
      'graph': g,
      'layer_name': layer_name,
    })

    self._nodes = list(set(self._nodes).union(g.nodes))
    self._nodes.sort()
  
  @property
  def num_nodes(self) -> int:
    """Get number of nodes"""
    if len(self.layers) == 0:
      return 0
    else:
      return len(self.nodes)
  
  @property
  def nodes(self) -> list:
    """Get list of nodes"""
    return self._nodes
  
  def sort_edges(self, edges, nodelist) -> list:
    sorted_edges = []

    for e in edges:
      if nodelist.index(e[0]) < nodelist.index(e[1]):
        sorted_edges.append((e[0],e[1]))
      else:
        sorted_edges.append((e[1],e[0]))
    
    sorted_edges.sort()

    return sorted_edges

  def get_sub_multiplex(self, nodes_to_keep: list):
    sub_mp = Multiplex

    for layer in self.layers:
      sg = layer['graph'].subgraph(nodes_to_keep)
      sub_mp.add_layer(sg, layer['layer_name'])
    
    return sub_mp

  def get_all_nodes_in_all_shortest_paths(self, nodes: list) -> list:
    output = set()

    for layer in self.layers:
      for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
          paths = nx.shortest_path(layer['graph'], node1, node2)

          for path in paths:
            output = output.union(set(path))
    
    return list(output)
