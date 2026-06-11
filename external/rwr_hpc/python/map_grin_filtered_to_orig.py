#!/usr/bin/env python3

import pandas as pd
import argparse

def map_grin_results(grin_filtered, orig_geneset, output_file):
  # Read in GRIN results
  df_grin = pd.read_csv(grin_filtered, sep='\t')

  # Read orig geneset
  df_orig = pd.read_csv(orig_geneset, sep='\t', header=None)

  df_out = df_orig[df_orig[1].isin(df_grin['INDEX'])]

  # Write output TSV
  df_out.to_csv(output_file, sep="\t", index=False, header=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Map GRIN filtering results to gene set")
  parser.add_argument("grin_filtered", help="tsv containing the GRIN filtering results")
  parser.add_argument("orig_geneset", help="tsv containing the GRIN input")
  parser.add_argument("output_file", help="Output file")

  args = parser.parse_args()

  map_grin_results(args.grin_filtered, args.orig_geneset, args.output_file)