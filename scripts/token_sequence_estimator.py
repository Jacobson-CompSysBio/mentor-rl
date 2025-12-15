from transformers import (
  AutoTokenizer
)
import json
import networkx as nx
import pandas as pd
import random
import math
import numpy as np

seed = 42
random.seed(seed)

flist="/lustre/orion/syb111/proj-shared/Personal/sullivanka/Data/Human_multiplex_networks/hnv3_ppi_tftarget_bulkscbrainpen_385layer_multiplex_flist.txt"
MODEL_NAME="gpt-oss-20b-bf16"
MODEL_PATH=f"/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/{MODEL_NAME}"

def get_node_and_layer_list(flist: str):
  node_list = []
  layer_list = []
  layer_info = pd.read_csv(flist, sep='\t', header=None)
  if layer_info.shape[1] < 2:
    raise ValueError('flist file must contain at least two tab seperated columns.')
  for i in layer_info.index:
    g = nx.read_edgelist(layer_info.iloc[i,0],create_using=nx.Graph,nodetype=str,data=(('weight', float),))
    layer_list.append(layer_info.iloc[i,1])
    node_list = set(node_list).union(g.nodes)

    # print(f'Density of layer {layer_info.iloc[i,1]}: {nx.density(g)}')
    print(f'Number of nodes and edges in layer {layer_info.iloc[i,1]}: {g.number_of_nodes()} {g.number_of_edges()}')
  
  print(f'There are {len(list(node_list))} unique nodes in the multiplex')
  return list(node_list), layer_list

def apply_chat_template(tokenizer, example, train=True):
  SYSTEM_PROMPT = (
    "You are a helpful biological chatbot. You will be given a biological question; "
    "return the correct answer."
  )
  
  messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": example["question"]},
    {"role": "assistant", "content": example["answer"]},
  ] if train else [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": example["question"]},
  ]
  
  if train:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
  else:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.pad_token_id = tokenizer.eos_token_id

# Verify pad_token_id is within vocabulary range
if tokenizer.pad_token_id >= tokenizer.vocab_size:
  print(f"[WARNING] pad_token_id ({tokenizer.pad_token_id}) >= vocab_size ({tokenizer.vocab_size}), setting to eos_token_id")
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.pad_token_id = tokenizer.eos_token_id

def get_num_tokens(example):
  formatted = apply_chat_template(tokenizer, example)
  tokens = tokenizer(formatted)
  num_tokens = len(tokens["input_ids"])
  return num_tokens

# Read in nodes and layer names from flist
node_list, layer_list = get_node_and_layer_list(flist)

def build_json(n_layers, n_nodes, n_edges):
  nodes = {}
  # Verify n_nodes <= len(node_list)
  if n_nodes > len(node_list):
    raise ValueError("n_nodes must be <= len(node_list)")
  
  # Randomly select `n_layers` from node_list
  sub_nodes = random.sample(node_list, k=n_nodes)

  # Add selected nodes to node_map
  for i, n in enumerate(sub_nodes):
    nid = f'N{i+1}'
    nodes[nid] = {"label": str(n)}
  
  layers = {}
  # Verify n_layers <= len(layer_list)
  if n_layers > len(layer_list):
    raise ValueError("n_layers must be <= len(layer_list)")
  
  # Randomly select `n_layers` from layer_list
  sub_layer = random.sample(layer_list, k=n_layers)

  # Add selected nodes to node_map
  for i, n in enumerate(sub_layer):
    lid = f'L{i+1}'
    layers[lid] = {"name": str(n)}

  edges = {}
  # Verify number of edges is less than `n_layers` * a complete graph worth of edges

  for i in range(0, n_edges):
    lid = f'L{i+1}'
    edge_list = []
    if i == 0:
      for e in range(1, n_edges):
        edge_list.append([f'N{e}', f'N{e+1}'])

    edges[lid] = edge_list

  doc = {
    "directed": False,
    "default_intra_weight": 1.0,
    "inter": {
      "type": "identity_full",
      "weight": 1.0,
      "switch_cost_default": 0.0
    },
    "layers": layers,
    "nodes": nodes,
    "edges": edges
  }

  return doc

# Base line 
data_dict = build_json(0, 0, 0)
json_str = json.dumps(data_dict)
example = {
  "question": f"Given the following multiplex, what is the topological distance between nodes ENSG00000185088 and ENSG00000156508? {json_str}",
  "answer": "1.0666666666666669"
}

num_tokens = get_num_tokens(example)
print(f'Baseline inputs took {num_tokens} tokens')


# Evalutate impact of L
# df = pd.DataFrame(None, columns=['avg_num_tokens'])
# for L in range(0, len(layer_list)+1):
#   num_tests = min([math.comb(len(layer_list), L), 30])

#   print(f'Performing {num_tests} tests for L={L}')

#   num_tokens_arr = list()
#   for t in range(num_tests):
#     data_dict = build_json(L, 0, 0)
#     json_str = json.dumps(data_dict)

#     if t == 0:
#       print(json_str)
#     example = {
#       "question": f"Given the following multiplex, what is the topological distance between nodes ENSG00000185088 and ENSG00000156508? {json_str}",
#       "answer": "1.0666666666666669"
#     }

#     num_tokens = get_num_tokens(example)

#     num_tokens_arr.append(num_tokens)
  
#   df.loc[len(df)] = [num_tokens_arr]

# df.to_csv('sequence_length_by_num_layers.tsv', sep='\t')


# Evalutate impact of N
# df = pd.DataFrame(None, columns=['avg_num_tokens'])
# for N in range(0, 1200):
#   num_tests = min([math.comb(len(node_list), N), 10])

#   print(f'Performing {num_tests} tests for N={N}')

#   num_tokens_arr = list()
#   for t in range(num_tests):
#     data_dict = build_json(0, N, 0)
#     json_str = json.dumps(data_dict)

#     if t == 0:
#       print(json_str)
#     example = {
#       "question": f"Given the following multiplex, what is the topological distance between nodes ENSG00000185088 and ENSG00000156508? {json_str}",
#       "answer": "1.0666666666666669"
#     }

#     num_tokens = get_num_tokens(example)

#     num_tokens_arr.append(num_tokens)
  
#   df.loc[len(df)] = [num_tokens_arr]

# df.to_csv('sequence_length_by_num_nodes.tsv', sep='\t')

# Evalutate impact of E
# df = pd.DataFrame(None, columns=['avg_num_tokens'])
# for E in range(0, 10000):
#   num_tests = min([math.comb(len(node_list), E), 1])
    
#   print(f'Performing {num_tests} tests for E={E}')

#   num_tokens_arr = list()
#   for t in range(num_tests):
#     data_dict = build_json(0, 0, E)
#     json_str = json.dumps(data_dict)

#     if t == 0:
#       print(json_str)
#     example = {
#       "question": f"Given the following multiplex, what is the topological distance between nodes ENSG00000185088 and ENSG00000156508? {json_str}",
#       "answer": "1.0666666666666669"
#     }

#     num_tokens = get_num_tokens(example)

#     num_tokens_arr.append(num_tokens)
  
#   df.loc[len(df)] = [num_tokens_arr]

# df.to_csv('sequence_length_by_num_edges_single_layer.tsv', sep='\t')


json_str = '{\"directed\": false, \"default_intra_weight\": 1.0, \"inter\": {\"type\": \"identity_full\", \"weight\": 1.0, \"switch_cost_default\": 0.0}, \"layers\": {\"L1\": {\"name\": \"layer1\"}, \"L2\": {\"name\": \"layer2\"}, \"L3\": {\"name\": \"layer3\"}}, \"nodes\": {\"N1\": {\"label\": \"node1\"}, \"N2\": {\"label\": \"node10\"}, \"N3\": {\"label\": \"node2\"}, \"N4\": {\"label\": \"node3\"}, \"N5\": {\"label\": \"node4\"}, \"N6\": {\"label\": \"node5\"}, \"N7\": {\"label\": \"node6\"}, \"N8\": {\"label\": \"node7\"}, \"N9\": {\"label\": \"node8\"}, \"N10\": {\"label\": \"node9\"}}, \"edges\": {\"L1\": [[\"N1\", \"N3\", 0.98], [\"N1\", \"N4\", 0.97], [\"N1\", \"N6\"], [\"N3\", \"N4\", 0.58], [\"N3\", \"N8\", 0.68], [\"N4\", \"N5\", 0.9], [\"N4\", \"N9\", 0.98], [\"N4\", \"N10\", 0.97], [\"N6\", \"N10\", 0.81], [\"N9\", \"N10\", 0.75], [\"N10\", \"N7\", 0.82]], \"L2\": [[\"N1\", \"N3\"], [\"N1\", \"N4\"], [\"N1\", \"N5\"], [\"N3\", \"N4\"], [\"N3\", \"N2\"], [\"N2\", \"N8\"]], \"L3\": [[\"N1\", \"N5\"], [\"N1\", \"N6\"], [\"N1\", \"N9\"], [\"N5\", \"N3\"], [\"N5\", \"N6\"], [\"N5\", \"N9\"], [\"N6\", \"N10\"], [\"N10\", \"N8\"]]}}' \

example = {
  "question": f"Given the following multiplex, what is the topological distance between nodes ENSG00000185088 and ENSG00000156508? {json_str}",
  "answer": "1.0666666666666669"
}

num_tokens = get_num_tokens(example)
print(f'num tokens: {num_tokens}')