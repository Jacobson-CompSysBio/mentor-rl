# import libraries
import argparse
import os
import pandas as pd
import numpy as np

# parse arguments
def parse_distance_qa_args():
  parser = argparse.ArgumentParser(description="Distance_QA_Generator")

  parser.add_argument('--distance_matrix',
                      type=str,
                      required=True,
                      help='Specify the file containing the distance matrix.')
  parser.add_argument('--output_dir',
                      type=str,
                      default='./',
                      help='Specify the directory in which to store the output file. If no value is provided the local directory will be used.')
  parser.add_argument('--outpur_file',
                      type=str,
                      default='distance_qas.tsv',
                      help='Specify the output file name. If no value is provided "distance_qas.tsv" will be used.')

  return parser.parse_args()

# Read and validate distance matrix
def read_distance_matrix(path: str) -> pd.DataFrame:
  """
  Reads a TSV distance matrix with one header row and one index column.
  Validates that it is either:
    1. symmetric, or
    2. has non-NaN values only in the upper triangle, or
    3. has non-NaN values only in the lower triangle.
  """
  df = pd.read_csv(path, sep='\t', index_col=0)

  if df.shape[0] != df.shape[1]:
    raise ValueError("Distance matrix must be square")

  if np.any(df.columns != df.index):
    raise ValueError("Column labels and row labels must match")

  arr = df.values
  ui = np.triu_indices(df.shape[0], 1)
  li = np.tril_indices(df.shape[0], -1)

  # Check for symmetry
  if np.allclose(arr, arr.T, equal_nan=True):
    return df

  # Check for upper-triangular pattern
  if np.all(np.isnan(arr[li])) and np.any(~np.isnan(arr[ui])):
    return df

  # Check for lower-triangular pattern
  if np.all(np.isnan(arr[ui])) and np.any(~np.isnan(arr[li])):
    return df.T

  raise ValueError("Distance matrix must be symmetric, or upper/lower-triangular with NaNs elsewhere")


# Create Q/A pairs
def write_qa_pairs(df: pd.DataFrame, path: str, file_name: str) -> None:
  os.makedirs(path, exist_ok=True)

  output_file = os.path.join(path, file_name)

  print(f'Expected output file: {output_file}')
  
  with open(output_file, 'w') as fp:
    fp.write('question\tlabel\n')

    labels = df.columns.to_list()
    N = len(labels)
    for i in range(N):
      node1 = labels[i]
      for j in range(i+1,N):
        node2 = labels[j]
        dist = df.iloc[i,j]
        fp.write( f'What is the topological distance between nodes {node1} and {node2}?\t{dist}\n')

def main():
  args = parse_distance_qa_args()

  dist_df = read_distance_matrix(args.distance_matrix)
  write_qa_pairs(dist_df, args.output_dir, args.outpur_file)

if __name__ == "__main__":
  main()
