import pytest
import os
import json
from src.memory.persistent_memory import PersistentMemory # Adjust if needed

# Fixture to provide a temporary memory file path for each test
@pytest.fixture
def temp_memory_file(tmp_path):
    file_path = tmp_path / "test_unit_memory.json"
    return str(file_path)

def test_memory_init_load_empty(temp_memory_file):
    """Test initialization when file doesn't exist."""
    mock_config = {'memory': {'persist_file': temp_memory_file}}
    memory = PersistentMemory(mock_config)
    assert memory.memory_data == {}
    assert not os.path.exists(temp_memory_file) # Shouldn't create file on init

def test_memory_update_save(temp_memory_file):
    """Test updating data saves to file."""
    mock_config = {'memory': {'persist_file': temp_memory_file}}
    memory = PersistentMemory(mock_config)
    user_id = "unit_user"

    memory.update_user_data(user_id, "name", "Unit Tester")
    memory.update_user_data(user_id, "preference", "fast tests")

    # Check file content
    assert os.path.exists(temp_memory_file)
    with open(temp_memory_file, 'r') as f:
        saved_data = json.load(f)
    assert user_id in saved_data
    assert saved_data[user_id]["name"] == "Unit Tester"
    assert saved_data[user_id]["preference"] == "fast tests"

def test_memory_load_existing(temp_memory_file):
    """Test loading data from a pre-existing file."""
    mock_config = {'memory': {'persist_file': temp_memory_file}}
    user_id = "load_user"
    # Pre-populate the file
    initial_data = {user_id: {"last_action": "tested loading"}}
    with open(temp_memory_file, 'w') as f:
        json.dump(initial_data, f)

    # Load memory
    memory = PersistentMemory(mock_config)
    assert user_id in memory.memory_data
    assert memory.memory_data[user_id]["last_action"] == "tested loading"

def test_memory_get_summary(temp_memory_file):
    """Test the summary generation."""
    mock_config = {'memory': {'persist_file': temp_memory_file}}
    memory = PersistentMemory(mock_config)
    user_id = "summary_user"
    memory.update_user_data(user_id, "name", "Summarizer")
    memory.update_user_data(user_id, "last_order_items", "Pixel Pasta")

    summary = memory.get_summary_for_prompt(user_id)
    assert "Summarizer" in summary
    assert "Pixel Pasta" in summary

    summary_empty = memory.get_summary_for_prompt("non_existent_user")
    assert "No previous interaction" in summary_empty
