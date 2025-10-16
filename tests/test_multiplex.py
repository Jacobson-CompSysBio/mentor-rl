import pytest
from pathlib import Path
import networkx as nx

from utils.multiplex import Multiplex

@pytest.fixture
def flist_path():
  return Path(__file__).parent / "fixtures" / "test_flist.flist"

@pytest.fixture
def test_multiplex(flist_path):
  return Multiplex(flist=str(flist_path))

def test_multiplex_init_layers(test_multiplex):
  assert len(test_multiplex) == 3
  assert test_multiplex[0]['layer_name'] == 'layer1'
  assert test_multiplex[1]['layer_name'] == 'layer2'
  assert test_multiplex[2]['layer_name'] == 'layer3'

def test_multiplex_nodes(test_multiplex):
  # Nodes across all layers should be unioned and sorted
  expected_nodes = ['node1','node10','node2','node3','node4','node5','node6','node7','node8','node9']
  assert test_multiplex.nodes == expected_nodes
  assert test_multiplex.num_nodes == 10

def test_multiplex_layer_graph_structure(test_multiplex):
  layer1_graph = test_multiplex[0]['graph']
  assert isinstance(layer1_graph, nx.Graph)
  assert layer1_graph.number_of_edges() == 11
  assert layer1_graph.has_edge('node2', 'node3')
  assert layer1_graph['node6']['node9']['weight'] == 0.82

  layer2_graph = test_multiplex[1]['graph']
  assert isinstance(layer2_graph, nx.Graph)
  assert layer2_graph.number_of_edges() == 6
  assert layer2_graph.has_edge('node2', 'node10')

  layer3_graph = test_multiplex[2]['graph']
  assert isinstance(layer3_graph, nx.Graph)
  assert layer3_graph.number_of_edges() == 8
  assert layer3_graph.has_edge('node5', 'node9')

def test_multiplex_to_json_default(tmp_path, test_multiplex):
  out_path = tmp_path / "output.json"
  json_doc = test_multiplex.to_json(path=out_path)

  assert out_path.exists()
  assert json_doc["directed"] is False
  assert "L1" in json_doc["edges"]
  assert "L2" in json_doc["edges"]
  assert len(json_doc["nodes"]) == 10
  assert json_doc["inter"]["type"] == "identity_full"
  assert json_doc["layers"]["L1"]["name"] == "layer1"

  # Check raw file content formatting
  content = out_path.read_text(encoding="utf-8")
    
  # Check that pretty formatting produced readable output
  # Each indent level adds 2 spaces (from indent=2)
  space_count = content.count("  ")  # 2-space indents
  newline_count = content.count("\n")
  
  assert space_count == 0
  assert newline_count == 0

  json_doc_pretty = test_multiplex.to_json(path=out_path, pretty=True)

  # Check raw file content formatting
  content = out_path.read_text(encoding="utf-8")
    
  # Check that pretty formatting produced readable output
  # Each indent level adds 2 spaces (from indent=2)
  space_count = content.count("  ")  # 2-space indents
  newline_count = content.count("\n")
  
  assert space_count > 0
  assert newline_count > 0

  assert json_doc == json_doc_pretty

def test_invalid_flist_raises(tmp_path):
  # Only one column = invalid
  bad_flist = tmp_path / "bad.tsv"
  bad_flist.write_text("tests/fixtures/layer1.edgelist\n")

  with pytest.raises(ValueError, match="flist file must contain at least two tab"):
      Multiplex(flist=str(bad_flist))

def test_empty_multiplex():
  m = Multiplex()
  assert len(m) == 0
  assert m.num_nodes == 0
  assert m.nodes == []
