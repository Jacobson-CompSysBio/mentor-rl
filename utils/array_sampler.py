import numpy as np
from numpy.random import Generator, PCG64

class array_sampler:
  def __init__(self, arr: np.ndarray) -> None:
    self.arr = np.array(arr)
    self.num_values = self.arr.size
    self.rng = Generator(PCG64())
  
  def sample(self):
    if self.num_values == 0:
      raise ValueError('All values from array have been selected')
    
    if self.num_values == 1:
      self.num_values -= 1
      return self.arr[0]

    idx = self.rng.choice(self.num_values-1)
    val = self.arr[idx]
    self.arr[idx] = self.arr[self.num_values-1]
    self.num_values -= 1

    return val

  @property
  def empty(self) -> bool:
    return self.num_values == 0