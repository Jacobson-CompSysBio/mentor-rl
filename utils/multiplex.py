# import libraries
import networkx as nx
import pandas as pd
import os
import json

class Multiplex:
  """Class for multiplex"""

  def __init__(self, flist: str = ''):
    self.layers = []
    self._nodes = []

    if flist:
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
  
  def to_json(self,
              path: str = '',
              pretty: bool = False,
              default_intra_weight:float = 1.0,
              inter_type:str = "identity_full",
              inter_weight:float = 1.0,
              switch_cost_default:float = 0.0) -> dict:
    """
    Write the multiplex to a JSON schema.

    Parameters
    ----------
    pretty: bool, optional

    path : str, optional
        If provided, write JSON to this file; otherwise return dict only.
    default_intra_weight : float
        Default weight to assign when edges lack 'weight' attribute.
    inter_type : str
        Inter-layer coupling type; usually 'identity_full'.
    inter_weight : float
        Default inter-layer weight.
    switch_cost_default : float
        Default cost for switching layers.

    Returns
    -------
    dict
        JSON-serializable dictionary of the multiplex.

    """

    # Map node IDs -> short IDs(N1, N2, N3, ...)
    nodes = {}
    node_index = {}
    for i, n in enumerate(self._nodes):
      nid = f'N{i+1}'
      nodes[nid] = {"label": str(n)}
      node_index[n] = nid

    # Layers
    layers = {f'L{i+1}': {'name': layer['layer_name']}
              for i, layer in enumerate(self.layers)}    
  
    # Edges
    edges = {}
    for i, layer in enumerate(self.layers):
      lid = f'L{i+1}'
      g = layer['graph']
      edge_list = []
      for u, v, data in g.edges(data=True):
        if u not in node_index or v not in node_index:
          continue

        w = data.get('weight', default_intra_weight)
        edge_list.append(
          [node_index[u], node_index[v], float(w)]
          if w != default_intra_weight else
          [node_index[u], node_index[v]]
        )
      edges[lid] = edge_list
    
    # Compose document
    doc = {
      "directed": False,
      "default_intra_weight": float(default_intra_weight),
      "inter": {
        "type": str(inter_type),
        "weight": float(inter_weight),
        "switch_cost_default": float(switch_cost_default)
      },
      "layers": layers,
      "nodes": nodes,
      "edges": edges
    }

    # Write JSON if a path is provided
    if path:
      dir_name = os.path.dirname(path)
      if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
      with open(path, "w", encoding="utf-8") as f:
        if pretty:
          json.dump(doc, f, ensure_ascii=False, indent = 2, separators=(",", ": "))
        else:
          json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))

    return doc
