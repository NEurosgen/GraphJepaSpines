import pytest

def pytest_addoption(parser):
    parser.addoption("--dataset_p", action="store", default=None, help="Path to the dataset")
    parser.addoption("--stats_p", action="store", default=None, help="Path to the stats directory")
    parser.addoption("--test_mode", action="store", default="single", choices=["single", "full"], help="Test mode: 'single' or 'full'")

@pytest.fixture
def dataset_p(request):
    return request.config.getoption("--dataset_p")

@pytest.fixture
def stats_p(request):
    return request.config.getoption("--stats_p")

@pytest.fixture
def test_mode(request):
    return request.config.getoption("--test_mode")
#python -m pytest tests/test_data_pipeline.py -s --dataset_p /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/graph_dataset_prepared --stats_p /home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/ --test_mode full