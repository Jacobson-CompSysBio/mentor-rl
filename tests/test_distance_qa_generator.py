import pytest
from unittest import mock
import pandas as pd
import numpy as np

from scripts.distance_qa_generator import parse_distance_qa_args, read_distance_matrix, write_qa_pairs

# helper function for tests
def write_tsv(df, tmp_path, name):
  path = tmp_path / name
  df.to_csv(path, sep='\t')
  return path

# test for required arguments
def test_distance_file_missing():
  # argparse raises SystemExit(2) when required args are missing
  with pytest.raises(SystemExit):
    parse_distance_qa_args()

# Tests for reading and valiating distanec matrix
def test_non_square_matrix(tmp_path):
  df = pd.DataFrame([[0, 1, 2],
                     [1, 0, 3]],
                  index=["A", "B"],
                  columns=["A", "B", "C"])
  path = write_tsv(df, tmp_path, "nonsquare.tsv")
  with pytest.raises(ValueError, match="Distance matrix must be square"):
    read_distance_matrix(path)

def test_label_mismatch_matrix(tmp_path):
  df = pd.DataFrame([[0, 1, 2],
                     [1, 0, 3],
                     [2, 3, 0]],
                  index=["A", "B", "C"],
                  columns=["A", "B", "D"])
  path = write_tsv(df, tmp_path, "symmetric.tsv")
  with pytest.raises(ValueError, match="Column labels and row labels must match"):
    read_distance_matrix(path)

def test_valid_symmetric_matrix(tmp_path):
  df = pd.DataFrame([[0, 1, 2],
                      [1, 0, 3],
                      [2, 3, 0]],
                    index=["A", "B", "C"],
                    columns=["A", "B", "C"])
  path = write_tsv(df, tmp_path, "symmetric.tsv")
  result = read_distance_matrix(path)
  pd.testing.assert_frame_equal(result, df)

def test_upper_triangle_only(tmp_path):
  df = pd.DataFrame([[np.nan, 1, 2],
                      [np.nan, np.nan, 3],
                      [np.nan, np.nan, np.nan]],
                    index=["A", "B", "C"],
                    columns=["A", "B", "C"])
  path = write_tsv(df, tmp_path, "upper.tsv")
  result = read_distance_matrix(path)
  pd.testing.assert_frame_equal(result, df)

def test_lower_triangle_only(tmp_path):
  df = pd.DataFrame([[0.0, np.nan, np.nan],
                      [1.0, 0, np.nan],
                      [2.0, 3.0, 0.0]],
                    index=["A", "B", "C"],
                    columns=["A", "B", "C"])
  path = write_tsv(df, tmp_path, "lower.tsv")
  result = read_distance_matrix(path)
  pd.testing.assert_frame_equal(result, df.T)

def test_invalid_asymmetric_matrix(tmp_path):
  df = pd.DataFrame([[0, 1, 2],
                      [9, 0, 3],
                      [2, 3, 0]],
                    index=["A", "B", "C"],
                    columns=["A", "B", "C"])
  path = write_tsv(df, tmp_path, "asymmetric.tsv")
  with pytest.raises(ValueError, match="Distance matrix must be symmetric"):
      read_distance_matrix(path)

def test_qa_from_lower_matrix(tmp_path):
  distance_matrix = './tests/fixtures/lower_matrix.tsv'
  
  # Create a temporary output directory
  out_dir = tmp_path / "output"
  file_name = 'distance_qas.tsv'

  dist_df = read_distance_matrix(distance_matrix)
  write_qa_pairs(dist_df, out_dir, file_name)

  # Read the output as a string
  output_file = out_dir / file_name
  with open(output_file, "r", encoding="utf-8") as f:
    result = f.read()

  expected_output=("question\tlabel\n"
                   "What is the topological distance between nodes geneA and geneB?\t0.1027979970619597\n"
                   "What is the topological distance between nodes geneA and geneC?\t0.0983246597894795\n"
                   "What is the topological distance between nodes geneA and geneD?\t0.124827069642217\n"
                   "What is the topological distance between nodes geneA and geneE?\t0.117903722944747\n"
                   "What is the topological distance between nodes geneA and geneF?\t0.1036542471175742\n"
                   "What is the topological distance between nodes geneB and geneC?\t0.1029951089169072\n"
                   "What is the topological distance between nodes geneB and geneD?\t0.1386619258139217\n"
                   "What is the topological distance between nodes geneB and geneE?\t0.1270672801100329\n"
                   "What is the topological distance between nodes geneB and geneF?\t0.0981850102193748\n"
                   "What is the topological distance between nodes geneC and geneD?\t0.1016590926674669\n"
                   "What is the topological distance between nodes geneC and geneE?\t0.1062875006279732\n"
                   "What is the topological distance between nodes geneC and geneF?\t0.0970199593917065\n"
                   "What is the topological distance between nodes geneD and geneE?\t0.1255450042519219\n"
                   "What is the topological distance between nodes geneD and geneF?\t0.1274651183666846\n"
                   "What is the topological distance between nodes geneE and geneF?\t0.1088391074981248\n")

  # Compare to expected string
  assert result == expected_output

def test_qa_from_upper_matrix(tmp_path):
  distance_matrix = './tests/fixtures/upper_matrix.tsv'
  
  # Create a temporary output directory
  out_dir = tmp_path / "output"
  file_name = 'distance_qas.tsv'

  dist_df = read_distance_matrix(distance_matrix)
  write_qa_pairs(dist_df, out_dir, file_name)

  # Read the output as a string
  output_file = out_dir / file_name
  with open(output_file, "r", encoding="utf-8") as f:
    result = f.read()

  expected_output=("question\tlabel\n"
                   "What is the topological distance between nodes geneA and geneB?\t0.1027979970619597\n"
                   "What is the topological distance between nodes geneA and geneC?\t0.0983246597894795\n"
                   "What is the topological distance between nodes geneA and geneD?\t0.124827069642217\n"
                   "What is the topological distance between nodes geneA and geneE?\t0.117903722944747\n"
                   "What is the topological distance between nodes geneA and geneF?\t0.1036542471175742\n"
                   "What is the topological distance between nodes geneB and geneC?\t0.1029951089169072\n"
                   "What is the topological distance between nodes geneB and geneD?\t0.1386619258139217\n"
                   "What is the topological distance between nodes geneB and geneE?\t0.1270672801100329\n"
                   "What is the topological distance between nodes geneB and geneF?\t0.0981850102193748\n"
                   "What is the topological distance between nodes geneC and geneD?\t0.1016590926674669\n"
                   "What is the topological distance between nodes geneC and geneE?\t0.1062875006279732\n"
                   "What is the topological distance between nodes geneC and geneF?\t0.0970199593917065\n"
                   "What is the topological distance between nodes geneD and geneE?\t0.1255450042519219\n"
                   "What is the topological distance between nodes geneD and geneF?\t0.1274651183666846\n"
                   "What is the topological distance between nodes geneE and geneF?\t0.1088391074981248\n")

  # Compare to expected string
  assert result == expected_output

def test_qa_from_symmetric_matrix(tmp_path):
  distance_matrix = './tests/fixtures/symmetric_matrix.tsv'
  
  # Create a temporary output directory
  out_dir = tmp_path / "output"
  file_name = 'distance_qas.tsv'

  dist_df = read_distance_matrix(distance_matrix)
  write_qa_pairs(dist_df, out_dir, file_name)

  # Read the output as a string
  output_file = out_dir / file_name
  with open(output_file, "r", encoding="utf-8") as f:
    result = f.read()

  expected_output=("question\tlabel\n"
                   "What is the topological distance between nodes geneA and geneB?\t0.1027979970619597\n"
                   "What is the topological distance between nodes geneA and geneC?\t0.0983246597894795\n"
                   "What is the topological distance between nodes geneA and geneD?\t0.124827069642217\n"
                   "What is the topological distance between nodes geneA and geneE?\t0.117903722944747\n"
                   "What is the topological distance between nodes geneA and geneF?\t0.1036542471175742\n"
                   "What is the topological distance between nodes geneB and geneC?\t0.1029951089169072\n"
                   "What is the topological distance between nodes geneB and geneD?\t0.1386619258139217\n"
                   "What is the topological distance between nodes geneB and geneE?\t0.1270672801100329\n"
                   "What is the topological distance between nodes geneB and geneF?\t0.0981850102193748\n"
                   "What is the topological distance between nodes geneC and geneD?\t0.1016590926674669\n"
                   "What is the topological distance between nodes geneC and geneE?\t0.1062875006279732\n"
                   "What is the topological distance between nodes geneC and geneF?\t0.0970199593917065\n"
                   "What is the topological distance between nodes geneD and geneE?\t0.1255450042519219\n"
                   "What is the topological distance between nodes geneD and geneF?\t0.1274651183666846\n"
                   "What is the topological distance between nodes geneE and geneF?\t0.1088391074981248\n")

  # Compare to expected string
  assert result == expected_output
