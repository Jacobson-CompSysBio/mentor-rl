import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------
## NORMAL DATASET
# ---------------
class BasicEdgePredDataset(Dataset):
  """
  Dataset for edge prediction with no .pt graph
  """

  def __init__(self, path: str):
    super().__init__()

    self.path = path
    self.text = pd.read_csv(f'{self.path}', sep='\t')

  def __len__(self):
    """Return the len of the dataset."""
    return len(self.text)
  
  def __getitem__(self, index: int):
    text = self.text.iloc[index]

    return {
      'id': index,
      'question': text['question'],
      'answer': text['label'],
      'desc': text['desc'],
    }
  
  def get_edgelist_text(self, desc):
     """
     Given a graph description, return the edgelist text, with edges separated by commas.
     """

     desc = desc.replace('[', '').replace(']', '').replace('\'', '')

     return desc

class PromptDataset(Dataset):

   def __init__(self, data):
       self.data = data

   def __len__(self):
       return len(self.data)

   def __getitem__(self, idx):
        return self.data[idx]

# ---------------
## GRPO MODS
# ---------------
class TransformedDataset(Dataset):
    def __init__(self, raw_dataset, transform):
        # Convert raw dataset to a list if it isn’t already
        self.raw_dataset = raw_dataset
        self.transform = transform

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        return self.transform(sample)
