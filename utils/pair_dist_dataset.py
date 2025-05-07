import torch
from torch.utils.data import Dataset
import pandas as pd

class GenePairDataset(Dataset):
  def __init__(self, file_name: str):
    super().__init__()

    self.file_name = file_name
    self.text = pd.read_csv(self.file_name, sep='\t')
  
  def __len__(self):
    """Return the length of the dataset."""
    return self.text.shape[1]
  
  def __getitem__(self, index: int):
    text = self.text.iloc[index]

    return {
      'id': index,
      'type': 'dist',
      'node1': text['node1'],
      'node2': text['node2'],
      'pos_test': text['pos_test'],
      'label': text['dist'],
      'edge_list': text['edge_list'],
    }
