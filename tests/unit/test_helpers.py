import pytest
import os
from src.utils.helpers import load_config # Assumes running pytest from root

def test_load_config_success(mock_config_path):
    """Tests loading a valid config file."""
    config = load_config(mock_config_path)
    assert isinstance(config, dict)
    assert "llm" in config
    assert config["llm"]["provider"] == "llama.cpp"
    assert config["active_business_type"] == "restaurant"

def test_load_config_not_found():
    """Tests that FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")
