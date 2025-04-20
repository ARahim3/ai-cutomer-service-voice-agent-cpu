# tests/integration/test_skills.py
import pytest
from pytest_mock import MockerFixture
# Use absolute imports from src assuming pytest runs from root
from src.skills.restaurant_skill import RestaurantSkill
from src.memory.persistent_memory import PersistentMemory

# Fixture to load the mock config from conftest.py
@pytest.fixture
def skill_config(mock_config_path):
     from src.utils.helpers import load_config
     return load_config(mock_config_path)

# Fixture providing an initialized skill with mocked dependencies
@pytest.fixture
def restaurant_skill_mocked(skill_config, mocker: MockerFixture):
    # Mock handlers needed by the skill
    mock_llm_handler = mocker.MagicMock()
    mock_rag_handler = mocker.MagicMock()
    # Create a real memory handler but mock its saving
    # Use a temporary file path for this test instance's memory
    temp_memory_path = skill_config['memory']['persist_file'] + ".skill_test"
    skill_config['memory']['persist_file'] = temp_memory_path # Modify config for test
    mock_memory_handler = PersistentMemory(skill_config)
    mocker.patch.object(mock_memory_handler, '_save_memory') # Prevent file save

    # Instantiate the skill
    skill = RestaurantSkill(skill_config, mock_llm_handler, mock_rag_handler, mock_memory_handler)

    # Mock the internal API call method within the skill instance
    mocker.patch.object(skill, '_make_mock_order_api_call')

    # Return the skill and dependencies for inspection
    return skill, mock_llm_handler, mock_rag_handler, mock_memory_handler

# --- Test Cases ---

def test_restaurant_place_order_success(restaurant_skill_mocked):
    """Test successful order placement."""
    skill, llm_mock, rag_mock, memory_mock = restaurant_skill_mocked
    # Configure the mock API call to return success
    skill._make_mock_order_api_call.return_value = {"success": True, "order_id": "mock_integ_123"}

    user_id = "integ_user_1"
    intent = "place_order"
    entities = {"items": [{"name": "Algorithm Burger", "quantity": 1}]}
    history = [{'role': 'user', 'content': 'order burger'}] # Minimal history

    response = skill.handle_intent(intent, entities, user_id, history)

    # Assertions
    assert "mock_integ_123" in response
    assert "Algorithm Burger" in response
    assert "placed successfully" in response
    skill._make_mock_order_api_call.assert_called_once_with(user_id, entities["items"])
    # Check if memory was updated (access mock directly - careful with impl details)
    assert memory_mock.memory_data[user_id]["last_order_id"] == "mock_integ_123"
    assert memory_mock.memory_data[user_id]["last_order_items"] == "1 Algorithm Burger"

def test_restaurant_place_order_api_failure(restaurant_skill_mocked):
    """Test order placement when mock API fails."""
    skill, _, _, memory_mock = restaurant_skill_mocked
    # Configure the mock API call to return failure
    skill._make_mock_order_api_call.return_value = {"success": False, "error": "Simulated network error"}

    user_id = "integ_user_2"
    intent = "place_order"
    entities = {"items": [{"name": "Data Delight Salad", "quantity": 1}]}
    history = [{'role': 'user', 'content': 'order salad'}]

    response = skill.handle_intent(intent, entities, user_id, history)

    assert "mock error placing your order" in response
    assert "Simulated network error" in response
    skill._make_mock_order_api_call.assert_called_once_with(user_id, entities["items"])
    # Ensure failed order ID wasn't saved
    assert memory_mock.memory_data.get(user_id, {}).get("last_order_id") is None

def test_restaurant_place_order_no_items(restaurant_skill_mocked):
    """Test place_order intent when LLM provides no item entities."""
    skill, _, _, _ = restaurant_skill_mocked
    user_id = "integ_user_3"
    intent = "place_order"
    entities = {} # No items extracted
    history = [{'role': 'user', 'content': 'I want to order'}]

    response = skill.handle_intent(intent, entities, user_id, history)

    assert "What items would you like" in response
    # Ensure API mock was NOT called
    skill._make_mock_order_api_call.assert_not_called()

# Add a test for a query intent (which should use the generic response path)
def test_restaurant_query_menu_uses_generic(restaurant_skill_mocked, mocker: MockerFixture):
     """Test that query_menu uses the response from history (simulating generic path)."""
     skill, llm_mock, rag_mock, memory_mock = restaurant_skill_mocked
     # Mock the get_generic_response method IF it's called directly by the skill
     # However, our current skill logic relies on the response already being in history
     # So, we just check that the API call is NOT made for this intent

     user_id = "integ_user_4"
     intent = "query_menu"
     entities = {}
     # Simulate history containing the LLM's response to the query
     history = [
         {'role': 'user', 'content': 'Tell me about the menu'},
         {'role': 'assistant', 'content': 'Our menu includes Pixel Pasta and Algorithm Burger.'}
     ]

     response = skill.handle_intent(intent, entities, user_id, history)

     # Assert that the response returned is the one from history
     assert response == 'Our menu includes Pixel Pasta and Algorithm Burger.'
     # Ensure API mock was NOT called for a query
     skill._make_mock_order_api_call.assert_not_called()
