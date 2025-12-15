# import libraries
import argparse
import os
import pandas as pd
import numpy as np
import json

# parse arguments
def parse_distance_qa_args():
  parser = argparse.ArgumentParser(description="Distance_QA_Generator")

  parser.add_argument('--distance_matrix',
                      type=str,
                      required=True,
                      help='Specify the file containing the distance matrix.')
  parser.add_argument('--multiplex',
                      type=str,
                      required=True,
                      help='JSON file representing the multiplex network.')
  parser.add_argument('--output_dir',
                      type=str,
                      default='./',
                      help='Specify the directory in which to store the output file. If no value is provided the local directory will be used.')
  parser.add_argument('--output_file',
                      type=str,
                      default='distance_qas.json',
                      help='Specify the output file name. If no value is provided "distance_qas.json" will be used.')

  return parser.parse_args()

# Validate arguments
def validate_arguments(args):
  if not os.path.exists(args.multiplex):
    raise ValueError('args.multiplex does not exist')

# Read and validate distance matrix
def read_distance_matrix(file_name: str) -> pd.DataFrame:
  """
  Reads a TSV distance matrix with one header row and one index column.
  Validates that it is either:
    1. symmetric, or
    2. has non-NaN values only in the upper triangle, or
    3. has non-NaN values only in the lower triangle.
  """
  df = pd.read_csv(file_name, sep='\t', index_col=0)

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

def get_json_to_str(json_file_name: str) -> str | None:
  try:
    # Open the JSON file in read mode
    with open(json_file_name, 'r') as file:
      data_dict = json.load(file)

    # Convert the dictionary back to a JSON formatted string
    json_string = json.dumps(data_dict)
 
    return json_string
  
  except FileNotFoundError:
    print(f'Error: The file {json_file_name} as not found.')
  except json.JSONDecodeError:
    print("EROR: Failed to decode the JSON from the file. Check for malformed JSON.")

# Create Q/A pairs
def write_qa_pairs(df: pd.DataFrame, out_dir: str, out_file_name: str, json_str: str | None) -> None:
  os.makedirs(out_dir, exist_ok=True)

  output_file = os.path.join(out_dir, out_file_name)
  
  # Add escape character to double quotes in json_str
  # json_str = json_str.replace('"', '"/')

  # Initialize an empty list to store the dictionaries
  qas = []

  # 
  labels = df.columns.to_list()
  N = len(labels)

  for i in range(N):
    node1 = labels[i]
    for j in range(N):
      if i == j:
        continue

      node2 = labels[j]

      if i < j:
        dist = df.iloc[i,j]
      else:
        dist = df.iloc[j,i]
      
      data_object = {
        "question": f"Given the following multiplex, what is the topological distance between nodes {node1} and {node2}? {json_str}",
        "answer": f"{dist}"
      }

      qas.append(data_object)
  
  # Write JSON to file
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(qas, f, ensure_ascii=False, indent = 2, separators=(",", ": "))

def main():
  args = parse_distance_qa_args()
  validate_arguments(args)

  dist_df = read_distance_matrix(args.distance_matrix)

  json_string = get_json_to_str(args.multiplex)
  write_qa_pairs(dist_df, args.output_dir, args.output_file, json_string)

if __name__ == "__main__":
  main()
