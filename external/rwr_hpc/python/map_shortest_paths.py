#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path

def remap_edges(shortest_paths_file, output_path):
    # Read TSV files
    df_edges = pd.read_csv(shortest_paths_file, sep="\t")

    # Keep and reorder columns
    df_edges = df_edges[["from", "to", "type", "weight"]]

    # Drop duplicate rows from TSV1
    n_lines = len(df_edges)
    df_edges = df_edges.drop_duplicates()

    if (len(df_edges) != n_lines):
        print(f'\tDropped {n_lines - len(df_edges)} duplicate edges')

    # Write output TSV
    df_edges.to_csv(output_path, sep="\t", index=False)

def find_and_process(dir_path: Path) -> None:
    if not dir_path.is_dir():
        print(f'skipping (not a directory): {dir_path}')
        return

    matches = list(dir_path.glob("*._shorest_paths.tsv"))

    if not matches:
        print(f'No matching files in {dir_path}')
        return

    for tsv_path in matches:
        remap_edges(tsv_path, dir_path)


def main():
    parser = argparse.ArgumentParser(description="Remap edge IDs to labels")
    parser.add_argument("shortest_paths_file", help="tsv containing all shortest path edges")
    parser.add_argument("output_file", help="Output file")
    parser.add_argument("--multiple_file", action="store_true", help="User provided a file of files to shortest paths")

    args = parser.parse_args()

    if (args.multiple_file):
        with args.shortest_paths_file.open() as f:
            for line in f:
                dir_path = Path(line.strip())
                if (dir_path):
                    find_and_process(dir_path)
    else:
        remap_edges(args.shortest_paths_file, args.output_file)



if __name__ == "__main__":
    main()
