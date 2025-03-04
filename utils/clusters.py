import pandas as pd

class Clusters:
  def __init__(self, file_name: str):
    self.data = pd.read_csv(file_name, sep='\t', index_col=1)

    if 'cluster' not in self.data.columns:
      raise ValueError(f"The column 'cluster' is missing from {file_name}")
  
  def __len__(self):
    """Returns the number of items in the cluster results"""
    return self.data.shape[0]

  def n_clusters(self):
    """Returns the number of clusters"""
    return len(self.data.cluster.unique())

  def get_cluster_nums(self):
    """Returns the uniqe cluster numbers"""
    return self.data.cluster.unique()
  
  def get_nodes_in_cluster(self, cluster_num):
    """Returns the nodes in the requested cluster"""
    if cluster_num not in self.data.cluster.values:
      raise KeyError(f'{cluster_num} is not a value cluster number')
    
    return self.data.index[self.data.cluster == cluster_num].tolist()
  
  def get_nodes(self):
    return self.data.index.tolist()
