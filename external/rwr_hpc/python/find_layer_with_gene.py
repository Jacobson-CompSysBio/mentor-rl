#!/usr/bin/env python3

import os
import pandas as pd
import argparse

def find_layers_with_gene(gene_set, gene_layer_map, output_dir):
  # Read TSV files
  df_geneset = pd.read_csv(gene_set, sep="\t", header=None)

  # Keep and reorder columns
  df_map = pd.read_csv(gene_layer_map, sep="\t", index_col=0)

  keys = pd.Index(df_geneset.iloc[:, 1])

  # present = keys.intersection(df_map.index)
  # missing = keys.difference(df_map.index)

  # missing_messages = [f"{k} not present in gene-to-layer" for k in missing]

  # # For present keys, extract columns with value == 1
  # results = (
  #   df_map.loc[present]
  #     .eq(1)
  #     .apply(lambda row: row.index[row].tolist(), axis=1)
  #     .to_dict()
  # )

  rows = []

  for key in keys:
    if key not in df_map.index:
      rows.append((key, "NOT_PRESENT", ""))
    else:
      cols = df_map.loc[key].eq(1)
      cols = cols[cols].index.tolist()
      rows.append((key, "PRESENT", ",".join(cols)))

  out_df = pd.DataFrame(rows, columns=["gene", "status", "columns_with_gene"])
  out_file = os.path.join(output_dir, "gene_to_layer_lookup_results.tsv")
  out_df.to_csv(out_file, sep="\t", index=False)


  # keep only keys that exist in df2.index
  valid_keys = df_map.index.intersection(keys)

  # count how many times each column == 1
  column_counts = df_map.loc[valid_keys].sum()

  # sort in decreasing order
  column_counts = column_counts.sort_values(ascending=False)

  # write to file (TSV)
  out_file = os.path.join(output_dir, "layer_count_for_mapped_genes.tsv")
  column_counts.to_csv(
    out_file,
    sep="\t",
    header=["count"]
)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Find which layers contain each gene in the input file")
  parser.add_argument("gene_set", help="Path to gene set")
  parser.add_argument("gene_layer_map", help="Path to gene-to-layer map. Expects rows to be genes and columns to be layers")
  parser.add_argument("output_dir", help="Output directory to save results")
  
  args = parser.parse_args()
  
  find_layers_with_gene(args.gene_set, args.gene_layer_map, args.output_dir)