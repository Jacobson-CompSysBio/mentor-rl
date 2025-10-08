# import libraries
import argparse
import os

# parse arguments
def parse_rwr_context_args():
  parser = argparse.ArgumentParser(description="RWR_Context_Generator")

  parser.add_argument('--output_dir',
                      type=str,
                      default='./',
                      help='Specify the directory in which to store the output file. If no value is provided the local directory will be used.')
  parser.add_argument('--outpur_file',
                      type=str,
                      default='rwr_context.txt',
                      help='Specify the output file name. If no value is provided "rwr_context.txt" will be used.')

  return parser.parse_args()

# print context to file
def print_context(file_name: str) -> None:
  context = (
    'For any nodes A and B, the topological distane between the nodes given a multiplex '
    'is equal to the spearman distance between two Random Walk with Restart(RWR) embeddings. '
    'The first embedding is calculated by performing RWR given an intial vector in which '
    'node A in each layer of the multiplex is equally weighted and all other values are zero. '
    'The second embedding is calulcated in a similar fashion. Before calculating the spearman '
    'distance, each vector is reduced across layers to a single value per node using the geometric '
    'mean.'
  )

  dir_name = os.path.dirname(file_name)
  if dir_name and not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)
  
  with open(file_name, "w") as fp:
    fp.writelines(context)

def main():
  args = parse_textualize_network_args()
  file_name = os.path.join(args.output_dir, args.outpur_file)
  print_context(file_name)

if __name__ == "__main__":
  main()
