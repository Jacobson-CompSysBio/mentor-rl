import sys
sys.path.append('../')
from utils.clusters import Clusters
import pytest

class TestClusters:
  @pytest.fixture
  def bad_clusters_filename(self):
    return 'badfilename.tsv'

  @pytest.fixture
  def bad_clusters_column_name(self):
    return 'tests/data/bad_clusters_column_name.tsv'
  
  @pytest.fixture
  def good_clusters_filename(self):
    return 'tests/data/clusters.tsv'
  
  # Test that the correct error is raised when a bad flist path is provided
  def test_bad_flist_path(self, bad_clusters_filename):
    with pytest.raises(FileNotFoundError):
      clusters = Clusters(bad_clusters_filename)
  
  # Test that the correct error is raised when a clusters file without 'cluster' is provided
  def test_bad_clusters_column_name(self, bad_clusters_column_name):
    with pytest.raises(ValueError):
      clusters = Clusters(bad_clusters_column_name)
  
  # Test that the correct error is raised when an invalid cluster number is provided
  def test_bad_cluster_num(self, good_clusters_filename):
    clusters = Clusters(good_clusters_filename)
    with pytest.raises(KeyError):
      clusters.get_nodes_in_cluster(0)
      clusters.get_nodes_in_cluster(4)
  
  # Test that the nodes in a cluster are correctly returned
  def test_good_cluster_num(self, good_clusters_filename):
    clusters = Clusters(good_clusters_filename)
    assert clusters.get_nodes_in_cluster(1) == ['ENSG00000185379']
    assert clusters.get_nodes_in_cluster(2) == ['ENSG00000104320','ENSG00000020922','ENSG00000113522']
    assert clusters.get_nodes_in_cluster(3) == ['ENSG00000108384','ENSG00000051180','ENSG00000083093']
  
  # Test that the number of clusters is correctly determined
  def test_n_clusters(self, good_clusters_filename):
    clusters = Clusters(good_clusters_filename)
    assert clusters.n_clusters() == 3

  # Test that the number of nodes in the cluster class is correctly determined
  def test_good_clusters_file(self, good_clusters_filename):
    clusters = Clusters(good_clusters_filename)

    assert len(clusters) == 7


