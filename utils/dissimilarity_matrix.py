import pandas as pd
import numpy as np

class DissimilarityMatrix:
  """Class for dissimilarity matrix"""

  def __init__(self, file_name: str = None):
    self.data = None
    self._nodes = []
    
    if file_name is not None:
      self.data = pd.read_csv(file_name, sep='\t', index_col=0)

      # Make sure matrix is square
      if self.data.shape[0] != self.data.shape[1]:
        raise ValueError('Dissimilarity matrix must be square')
      
      # Make sure index values and column values match
      if self.data.index.tolist() != self.data.columns.tolist():
        raise ValueError('Index values and column values in dissimilarity matrix must match')
      
      # Check that the matrix is either lower triangular, upper triangular, or symmetric

      self._nodes = self.data.index.tolist()
  
  def _get_dist(self, node_name: str) -> pd.Series:
    dist = self.data.loc[node_name]
    dist[dist.isna()] = self.data.loc[:,node_name]

    return dist

  @property
  def nodes(self) -> list:
    """Get list of nodes"""
    return self._nodes

  def get_dist(self, node_name1: str, node_name2: str):
    """Return the distance value"""
    if node_name1 not in self.data.index.values:
      raise KeyError(f'{node_name1} is not a valid key for the dissimilarity matrix')
    if node_name2 not in self.data.index.values:
      raise KeyError(f'{node_name2} is not a valid key for the dissimilarity matrix')
    return self.data.loc[node_name1, node_name2]
  
  def get_node_at_largest_dist(self, node_name: str) -> str:
    """Return the node with the largest distance 'node_name'"""
    dist = self._get_dist(node_name)
    return dist.idxmax()
  
  def get_largest_dist(self, node_name: str):
    """Return the largest distance between 'node_name' and any other node"""
    dist = self._get_dist(node_name)
    return max(dist)

  def get_all_distance_gte(self, threshold):
    out = pd.DataFrame(None, columns=['node1', 'node2', 'dist'])

    n = len(self.nodes)
    for i in range(n):
      for j in range(i):
        if self.data.iloc[i,j] >= threshold:
          out.loc[len(out)] = [self.nodes[i], self.nodes[j], self.data.iloc[i,j]]
    
    return out

  def get_all_pairwise_distances(self, nodes: list):
    out = pd.DataFrame(None, columns=['node1', 'node2', 'dist'])

    if (len(nodes) < 2):
      return out

    tmp_data = self.data.loc[nodes, nodes]
    n = len(nodes)
    for i in range(n):
      for j in range(i):
        out.loc[len(out)] = [tmp_data.index[i], tmp_data.columns[j], tmp_data.iloc[i,j]]
    
    return out
  