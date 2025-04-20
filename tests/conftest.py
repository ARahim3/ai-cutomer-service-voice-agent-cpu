import pytest
import os

@pytest.fixture(scope="session") # Available to all tests in the session
def mock_config_path():
    """Returns the path to the dummy test config file."""
    # Construct path relative to this conftest.py file
    return os.path.join(os.path.dirname(__file__), "fixtures", "test_config.yaml")

@pytest.fixture
def temp_file_path(tmp_path):
    """Fixture to provide a temporary file path within pytest's temp dir."""
    return tmp_path / "temp_test_file.data"
