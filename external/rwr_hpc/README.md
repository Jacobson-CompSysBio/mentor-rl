# RWR HPC
_RWR HPC_ is a C++ based software suite for network analysis, primarily using Random Walk with Restart (RWR). The suite contains five application which are executed through the command line interface.

# Apps
The software suite contains 5 applications / programs with source code located at `<rwr_hpc>/apps`. Descriptions of the prograns and command line argurments are listed below.

## RWR++
The _RWR_ application is a wrapper for the RWR library. It is used to calculate RWR encodings, rank the elements of each encoding, and calculate a distance matrix from the encodings. 

### Inputs
#### Required
`--flist`: Path to the file containing paths to and names of each layer in the multiplex.

#### Optional
`--no_edgelist_headers`: Flag that indicates that the edge lists contains no headers. This flag applies to all edge lists referenced by the `--flist`.

`--seed_file`: Tab delimited file containing seeds to initalize RWR probabilty vectors with. Seeds on a single line will be combined in a single vector. Seeds on mulitple lines will be encoded in seperate vectors. Any empty value will treat each node in the multiplex as a seed speately.

`--no_set_ids`: Indicates that the seed_file contains no set ids. These ids are stored as the first value in each row.

`--output_dir`: Output directory

`--restart`: Probabilty of restart during random walk. Defaults to 0.7.

`--delta`: Probabilty of jumping from a layer to any other layer in the mulitplex. Defaults to 0.5.

`--reduction_method`: The method used to reduce RWR embeddings to a single value per node. Defaults to geometric mean.

`--threshold`: Threshold used to determine when RWR converges. Defaults to 1e-10.

`--record_encodings`: Flag that indicates the RWR encodings should be recorded.

`--record_ranks`: Flag that indicates the ranked RWR encodings should be recorded.

`--distance_metric`: Metric used to calculate distance between all pairs of RWR encodings. Defaults to spearman.

`--record_nodes_by_layer`: Flag that indicates a file should be created recording which nodes are present in each layer.

`--filter_method`: Flag that indicates a that the RWR encodings should be filered before the distance matrix is calculated.

### Outputs
The distance matrix will be recorded at `<output_dir><distance_metric>_dist_matrix.tsv`.

If the `--record_ranks` flag is set the ranks will be recorded at `<output_dir>ranks.tsv`.

If the `--record_encodings` flag is set the encodings will be recorded at `<output_dir>encodings.tsv`.

If the `--record_nodes_by_layer` flag is set, nodes present in each layer will be recorded at `<output_dir>nodes_by_layer.tsv`.

## GRIN++
The _GRIN++_ program operates in two parts. The C++ code generates the Leave-One-Out (LOO) encodings for the genes in the input gene set and the mean LOO encodings for the null distribution. These encodings are then provided as inputs to the R script which filters the input genes.

### C++ Inputs
#### Required
`--flist`: Path to the file containing paths to and names of each layer in the multiplex.

`--seed_file`: Tab delimited file containing seeds to initalize RWR probabilty vectors with. Seeds on a single line will be combined in a single vector. Seeds on mulitple lines will be encoded in seperate vectors.

#### Optional
`--no_edgelist_headers`: Flag that indicates that the edge lists contains no headers. This flag applies to all edge lists referenced by the `--flist`.

`--no_set_ids`: Indicates that the seed_file contains no set ids. These ids are stored as the first value in each row.

`--output_dir`: Output directory

`--restart`: Probabilty of restart during random walk. Defaults to 0.7.

`--delta`: Probabilty of jumping from a layer to any other layer in the mulitplex. Defaults to 0.5.

`--reduction_method`: The method used to reduce RWR embeddings to a single value per node. Defaults to geometric mean.

`--threshold`: Threshold used to determine when RWR converges. Defaults to 1e-10.

`--record_encodings`: Flag that indicates the RWR encodings should be recorded.

`--record_ranks`: Flag that indicates the ranked RWR encodings should be recorded.

`--n_samples_null_dist`: "Number of random seeds sets to use to create null distribution. Defaults to 100.

`--seed`: Value used to seed random number generator for null distribution.

### R Script Inputs
#### Required
`--gene_ranks`: Path to the gene_ranks.tsv file from GRIN++.

`--null_ranks`: Path to the null_ranks.tsv file from GRIN++.

#### Optional
`--modname`: Alias for this run. Useful for output.

`--plot`: Include this parameter if you want to output PNG plots of results.

`--outdir`: Path to the output directory

`--threads`: Number of threads to use. default for your system is all cores - 1.

`--simple-filenames`: Use simple filenames.

`--verbose`: Log more stuff.

### Outputs
The C++ code generates two files: `<output_dir>gene_ranks.tsv` and `<output_dir>null_ranks.tsv`. 

The R script generates a file for the retained genes `<outdir>GRIN__<modname>__Retained_Gene.txt` and a file for the removed genes `<outdir>GRIN__<modname>__Removed_Gene.txt`. 

## Layer Ablation++
_Layer Ablation_ calculates the impact each layer has to the RWR embedding value of each seed. First, the RWR embeddings of each seed are calculated. Then, each layer is iteratively removed from the multiplex and a new RWR embedding is calculate based on each seed. The distance between the base embedding and the ablated embedding for each layer and seed are calculated.

### Inputs
#### Required
`--flist`: Path to the file containing paths to and names of each layer in the multiplex.

#### Optional
`--no_edgelist_headers`: Flag that indicates that the edge lists contains no headers. This flag applies to all edge lists referenced by the `--flist`.

`--seed_file`: Tab delimited file containing seeds to initalize RWR probabilty vectors with. Seeds on a single line will be combined in a single vector. Seeds on mulitple lines will be encoded in seperate vectors. Any empty value will treat each node in the multiplex as a seed separately.

`--no_set_ids`: Indicates that the pertubation_file and seed_file contains no set ids. These ids are stored as the first value in each row.

`--output_dir`: Output directory

`--restart`: Probabilty of restart during random walk. Defaults to 0.7.

`--delta`: Probabilty of jumping from a layer to any other layer in the mulitplex. Defaults to 0.5.

`--reduction_method`: The method used to reduce RWR embeddings to a single value per node. Defaults to geometric mean.

`--threshold`: Threshold used to determine when RWR converges. Defaults to 1e-10.

`--distance_metric`: Metric used to calculate distance between base embedding and abaltion embedding for all seeds. Defaults to spearman.

### Outputs
The abaltion distance matrix is recorded at `<output_dir><distance_metric>_ablation_distance_matrix.tsv`. The rows indicated the seed vector and the columns indicate the ablated layers.

## Node Perturbation++
_Node Perturbation_ calculates the impact each node has on the RWR embedding value of each seed. First, the RWR embeddings of each seed are calculated. Then, each node in the `--pertubation_file` is iteratively removed from the multiplex and a new RWR embedding is calculate based on each seed. The distance between the base embedding and the perturbed embedding for each node and seed are calculated.

### Inputs
#### Required
`--flist`: Path to the file containing paths to and names of each layer in the multiplex.

#### Optional
`--no_edgelist_headers`: Flag that indicates that the edge lists contains no headers. This flag applies to all edge lists referenced by the `--flist`.

`--pertubation_file`: Tab delimited file containing nodes to perturbate. Nodes on a single line will be perturbed together. Nodes on mulitple lines will be perturbed separately. Any empty value will perturbe each node in the multiplex separately.

`--seed_file`: Tab delimited file containing seeds to initalize RWR probabilty vectors with. Seeds on a single line will be combined in a single vector. Seeds on mulitple lines will be encoded in seperate vectors. Any empty value will treat each node in the multiplex as a seed separately.

`--no_set_ids`: Indicates that the pertubation_file and seed_file contains no set ids. These ids are stored as the first value in each row.

`--output_dir`: Output directory

`--restart`: Probabilty of restart during random walk. Defaults to 0.7.

`--delta`: Probabilty of jumping from a layer to any other layer in the mulitplex. Defaults to 0.5.

`--reduction_method`: The method used to reduce RWR embeddings to a single value per node. Defaults to geometric mean.

`--threshold`: Threshold used to determine when RWR converges. Defaults to 1e-10.

`--distance_metric`: Metric used to calculate distance between base encoding and perburbation encoding for all seeds. Defaults to spearman.

### Outputs
The perturbation distance matrix is recorded at `<output_dir><distance_metric>_perturbation_distance_matrix.tsv`. The rows indicated the seed vector and the columns indicate the perturbed node.

## All Shortest Paths
_All Shortest Paths_ calculates all shortest paths between each node in a `source` set and each node in a `target` set. If no `target` set is provided, the `source` set is used as the `target` set also. First, the edges in the multipex are merged together into a single layer based on the `merge_method`. Next, all shortest paths are found, either based on edge weight or path length. If by edge weight, each edge is converted to a distance before finding the shortest paths. After all paths have been found on the merged network, the _best_ version of each path element is idenfied in the multiplex. Finally, the layer-specific shortest paths and layer statistic files are recoreded.

### Inputs
#### Required
`--flist`: Path to the file containing paths to and names of each layer in the multiplex.

`--sources_file`: Tab delimited file containing nodes that act as shortest paths sources

#### Optional
`--no_edgelist_headers`: Flag that indicates that the edge lists contain no headers. This flag applies to all edge lists referenced by the `--flist`.

`--targets_file`: Tab delimited file containing nodes that act as shortest paths targets. If no targets file is provided sources will act as targets.

`--no_set_ids`: Indicates that the sources and targets files contains no set ids. These ids are stored as the first value in each row.

`--merge_method`: Method used to merge layers in multiplex. Defaults to `max` if not provided.

`--output_dir`: Output directory

`--run_tag`: Name pre-pended to output_files specifiying run.

`--ignore_weights`: Falg that indicates shortest paths should be found based on path length. This flag ignores any edge weights.

### Output
The shortest paths algorithm creates two output files. The first file, recorded at `<output_dir>/<run_tag>_shortest_paths.tsv`, details each edge of every shortest path. This information includes the `source`, `target`, and `weight` of each edge, along with the `layer` in the multiplex in which it is located. Additionaly, information about the shortest path on which the edge is found, shortest path length (number of edges), and all elements in the shortest path are included.

The second file, recorded at `<output_dir>/<run_tag>_layer_counts.tsv`, provides a list of layers and the number of shortest paths edges each layer contains. This is is sorted on descending order. Any layer names not printed contained no shortest path edges.

# Scripts
Skeleton scripts for recompiling the code and runnig each app are located at `<rwr_hpc>/scripts`.

The skeleton scripts are set up to run on a single node.
