#!/usr/bin/env python3

import os
import argparse
import csv
from collections import defaultdict

def split_clusters(input_tsv, output_dir, runtag):
  # Dictionary to hold clusters: {cluster_id: [labels]}
  clusters = defaultdict(list)

  # Read the TSV
  with open(input_tsv, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
      cluster_id = row['cluster']
      label = row['label']
      clusters[cluster_id].append(label)

  # Make sure output directory exists
  os.makedirs(output_dir, exist_ok=True)

  all_sources = []
  all_runtags = []
  all_output_dirs = []

  # Write one TSV per cluster
  for cluster_id, labels in clusters.items():
    # Make subdirectory
    sub_path = os.path.join(output_dir, f"module_{cluster_id}")
    os.makedirs(sub_path, exist_ok=True)

    local_run_tag = f'{runtag}_module{cluster_id}'

    out_file = os.path.join(sub_path, f"{local_run_tag}_seeds.tsv")
    
    with open(out_file, 'w') as f:
      writer = csv.writer(f, delimiter='\t', lineterminator="\n")
      for label in labels:
        writer.writerow([local_run_tag, label])

      if len(labels) > 1:
        all_sources.append(f'{out_file}')
        all_runtags.append(f'{local_run_tag}')
        all_output_dirs.append(f'{sub_path}')

  # Write accumlated files
  all_sources_file = os.path.join(output_dir, "all_sources.tsv")
  with open(all_sources_file, 'w') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator="\n")
    for source in all_sources:
      writer.writerow([source])
    
  all_runtags_file = os.path.join(output_dir, "all_runtags.tsv")
  with open(all_runtags_file, 'w') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator="\n")
    for tag in all_runtags:
      writer.writerow([tag])
    
  all_output_dirs_file = os.path.join(output_dir, "all_output_dirs.tsv")
  with open(all_output_dirs_file, 'w') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator="\n")
    for dir in all_output_dirs:
      writer.writerow([dir])

  print(f"Processed {len(clusters)} clusters. Output saved to '{output_dir}'.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Split TSV by cluster")
  parser.add_argument("input_tsv", help="Path to input TSV file")
  parser.add_argument("output_dir", help="Directory to save cluster TSVs")
  parser.add_argument("runtag", help="Runtag to add to output files")
  
  args = parser.parse_args()
  
  split_clusters(args.input_tsv, args.output_dir, args.runtag)
