# src/main.py
import sys
import os
import warnings
# # Adjust path to import from sibling directories
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print("--- DEBUG INFO ---")
# print("Running main.py")
# print("Current Working Directory:", os.getcwd())
# # Print the directory containing main.py itself
# print("main.py Directory:", os.path.dirname(os.path.abspath(__file__)))
# # Print the sys.path Python uses for imports
# print("sys.path:")
# for p in sys.path:
#     print(f"  - {p}")
# print("--- END DEBUG INFO ---")

warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

from .utils.helpers import load_config
from .core.conversation_manager import ConversationManager

def run_agent():
    """Loads config and starts the conversation manager."""
    try:
        config = load_config() # Loads from src/config.yaml by default
        manager = ConversationManager(config)
        manager.start_conversation()
    except FileNotFoundError as e:
         print(f"Fatal Error: {e}")
         print("Ensure config.yaml exists in the 'src' directory.")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         # Add more detailed logging here if needed
         import traceback
         traceback.print_exc()

if __name__ == "__main__":
    print("Starting AI Agent...")
    run_agent()
    print("AI Agent finished.")