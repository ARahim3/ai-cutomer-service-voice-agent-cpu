import json
import os
from datetime import datetime

class PersistentMemory:
    def __init__(self, config):
        self.memory_file = config['memory'].get('persist_file', 'user_memory.json')
        print(f"Initializing Persistent Memory (File: {self.memory_file})...")
        self.memory_data = self._load_memory()
        print("Persistent Memory initialized.")

    def _load_memory(self):
        """Loads memory data from the JSON file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Memory file '{self.memory_file}' is corrupted. Starting fresh.")
                return {}
            except Exception as e:
                 print(f"Warning: Could not load memory file '{self.memory_file}': {e}. Starting fresh.")
                 return {}
        else:
            print("Memory file not found. Starting fresh.")
            return {}

    def _save_memory(self):
        """Saves the current memory data to the JSON file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory_data, f, indent=4)
        except Exception as e:
            print(f"Error saving memory file '{self.memory_file}': {e}")

    def get_user_data(self, user_id):
        """Retrieves data for a specific user."""
        return self.memory_data.get(user_id, {})

    def update_user_data(self, user_id, key, value):
        """Updates a specific key for a user and saves memory."""
        if user_id not in self.memory_data:
            self.memory_data[user_id] = {}
        self.memory_data[user_id][key] = value
        # Include a timestamp for the update
        self.memory_data[user_id]['last_updated'] = datetime.now().isoformat()
        self._save_memory()

    def get_summary_for_prompt(self, user_id, max_len=200):
        """Gets a concise summary of user data suitable for an LLM prompt."""
        user_data = self.get_user_data(user_id)
        if not user_data:
            return "No previous interaction history found for this user."

        summary_parts = []
        if 'name' in user_data:
             summary_parts.append(f"User's name might be {user_data['name']}.")
        if 'last_order_items' in user_data:
            summary_parts.append(f"Last order included: {user_data['last_order_items']}.")
        if 'preferences' in user_data:
            summary_parts.append(f"Preferences: {user_data['preferences']}.")
        # Add more fields as needed

        summary = " ".join(summary_parts)
        if len(summary) > max_len:
             summary = summary[:max_len] + "..."
        return summary if summary else "Some previous interaction history exists."
