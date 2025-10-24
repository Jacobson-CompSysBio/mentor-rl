import pytest
from unittest import mock
import sys
import os
 
from scripts.rwr_context_generator import parse_rwr_context_args, print_context

# helper to mock sys.argv and call parse
def parse_args_with(argv):
  with mock.patch.object(sys, 'argv', argv):
    return parse_rwr_context_args()

# test default arguments
def test_default_args():
  args = parse_rwr_context_args()

  assert args.output_dir == './'
  assert args.outpur_file == 'rwr_context.txt'

def test_nondefault_args():
  args = parse_args_with(['prog', '--output_dir', '/my/dir', '--outpur_file', 'my_context.txt'])

  assert args.output_dir == '/my/dir'
  assert args.outpur_file == 'my_context.txt'

def test_print_context(tmp_path):
  output_file = os.path.join(tmp_path, 'output.json')
  print_context(output_file)

  with open(output_file, "r", encoding="utf-8") as f:
    result = f.read()

  expected_result = (
    'For any nodes A and B, the topological distane between the nodes given a multiplex '
    'is equal to the spearman distance between two Random Walk with Restart(RWR) embeddings. '
    'The first embedding is calculated by performing RWR given an intial vector in which '
    'node A in each layer of the multiplex is equally weighted and all other values are zero. '
    'The second embedding is calulcated in a similar fashion. Before calculating the spearman '
    'distance, each vector is reduced across layers to a single value per node using the geometric '
    'mean.'
  )

  assert expected_result == result
