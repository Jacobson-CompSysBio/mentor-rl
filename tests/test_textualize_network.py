import pytest
from unittest import mock
import sys

from scripts.textualize_network import parse_textualize_network_args, create_full_output_file_name

# helper to mock sys.argv and call parse
def parse_args_with(argv):
  with mock.patch.object(sys, 'argv', argv):
    return parse_textualize_network_args()

# test for required arguments
def test_flist_missing():
  # argparse raises SystemExit(2) when required args are missing
  with pytest.raises(SystemExit):
    parse_textualize_network_args()

# test output dir
def test_output_dir_default():
  args = parse_args_with(['prog', '--flist', 'networks.tsv'])
  assert args.output_dir == './'  # default value

def test_output_dir_nodefault():
  args = parse_args_with(['prog', '--flist', 'networks.tsv', '--output_dir', '/my/path'])
  assert args.output_dir == '/my/path'

# test output file
def test_output_file_default():
  args = parse_args_with(['prog', '--flist', 'networks.tsv'])
  assert args.output_file == ''  # default value

def test_output_file_nodefault():
  args = parse_args_with(['prog', '--flist', 'networks.tsv', '--output_file', 'my_file_name.txt'])
  assert args.output_file == 'my_file_name.txt'

# test pretty_printing
def test_pretty_printing_default():
  args = parse_args_with(['prog', '--flist', 'networks.tsv'])
  assert args.pretty_printing == False  # default value

def test_pretty_printing_set():
  args = parse_args_with(['prog', '--flist', 'networks.tsv', '--pretty_printing'])
  assert args.pretty_printing == True  # set value

# test creation of full output nam
def test_create_full_output_file_name_default():
  output_dir = './'
  output_file = ''
  flist = 'my_flist.tsv'

  full_file_name = create_full_output_file_name(output_dir, output_file, flist)
  assert full_file_name == './my_flist.json'

def test_create_full_output_file_name_nodefault():
  output_dir = '/my/path'
  output_file = 'my_textualized_network.txt'
  flist = 'my_flist.tsv'

  full_file_name = create_full_output_file_name(output_dir, output_file, flist)
  assert full_file_name == '/my/path/my_textualized_network.txt'

