# import libraries
import argparse
import os
from utils.multiplex import Multiplex

# parse arguments
def parse_textualize_network_args():
  parser = argparse.ArgumentParser(description="Textualize_Network")

  parser.add_argument('--flist',
                      type=str,
                      required=True,
                      help='Specify the path to the input flist, a tab seperate documents contining paths to networks and thier names.')
  parser.add_argument('--output_dir',
                      type=str,
                      default='./',
                      help='Specify the directory in which to store the output file. If no value is provided the local directory will be used.')
  parser.add_argument('--output_file',
                       type=str,
                       help='Specify the name of the output file. If no value is provided the name of the flist file will be used.',
                       default='')
  parser.add_argument('--pretty_printing',
                      action='store_true',
                      help='Enable pretty-printing of the JSON output. Otherwise output will be compact.')
  return parser.parse_args()

# create full output file name
def create_full_output_file_name(output_dir: str, output_file: str, flist: str) -> str:
  if output_file == '':
    output_file = f'{os.path.splitext(os.path.basename(flist))[0]}.json'

  return os.path.join(output_dir, output_file)

def main():
  args = parse_textualize_network_args()

  full_output_file = create_full_output_file_name(args.output_dir, args.output_file, args.flist)

  mp = Multiplex(args.flist)
  mp.to_json(full_output_file, args.pretty_printing)

if __name__ == "__main__":
  main()
  