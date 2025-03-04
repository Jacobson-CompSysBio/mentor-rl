import sys

sys.path.append('../')

from utils.multiplex import Multiplex
import pytest
import networkx as nx

class TestMultiplex:
  @pytest.fixture
  def bad_filenae_flist(self):
    return 'badfilename.flist'
    
  @pytest.fixture
  def bad_format_flist(self):
    return 'tests/data/bad_format.flist'
    
  @pytest.fixture
  def bad_graph_path_flist(self):
    return 'tests/data/bad_graph_path.flist'
    
  @pytest.fixture
  def monoplex_flist(self):
    return 'tests/data/monoplex.flist'
    
  @pytest.fixture
  def multiplex_flist(self):
    return'tests/data/multiplex.flist'
  
  # Test multiplex object when no flist is provided
  def test_no_flist(self):
    mp = Multiplex()

    assert mp.layers == []
    assert mp._nodes == []
    assert len(mp) == 0
    assert mp.num_nodes == 0
    assert mp.nodes == []

    with pytest.raises(IndexError):
      assert mp[0]

  # Test that the correct error is raised when a bad flist path is provided
  def test_bad_flist_path(self, bad_filenae_flist):
    with pytest.raises(FileNotFoundError):
      mp = Multiplex(bad_filenae_flist)

  # Test that the correct error is raised when a flist with bad format is provided
  def test_flist_bad_format(self, bad_format_flist):
    with pytest.raises(ValueError):
      mp = Multiplex(bad_format_flist)

  # Test that the correct error is raised when a graph path does not exist
  def test_bad_graph_path(self, bad_graph_path_flist):
    with pytest.raises(FileNotFoundError):
      mp = Multiplex(bad_graph_path_flist)

  def test_monoplex_flist(self, monoplex_flist):
    mp = Multiplex(monoplex_flist)

    assert len(mp) == 1
    assert isinstance(mp[0]['graph'], nx.Graph)
    assert mp[0]['layer_name'] == 'type_1'

    assert mp.nodes == ['G1','G10','G2','G3','G4','G5','G6','G7','G8','G9']
    assert mp.nodes == mp._nodes
    assert mp.num_nodes == 10

  def test_multiplex_flist(self, multiplex_flist):
    mp = Multiplex(multiplex_flist)

    assert len(mp) == 5
    assert mp[0]['layer_name'] == 'type_1'
    assert mp[1]['layer_name'] == 'type_2'
    assert mp[2]['layer_name'] == 'type_3'
    assert mp[3]['layer_name'] == 'type_4'
    assert mp[4]['layer_name'] == 'type_5'

    assert mp.nodes == ['G1','G10','G2','G3','G4','G5','G6','G7','G8','G9']
    assert mp.nodes == mp._nodes
    assert mp.num_nodes == 10

