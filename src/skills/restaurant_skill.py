from .base_skill import BaseSkill
import json
import random # For mock API simulation
import os # If using env vars for API keys/URLs later

class RestaurantSkill(BaseSkill):
    skill_id = "restaurant"

    def _make_mock_order_api_call(self, user_id, items):
        """Simulates calling the restaurant order API."""
        print(f"MOCK API CALL: Placing order for user {user_id}, items: {items}")
        if not items: return {"success": False, "error": "No items provided"}
        # Simulate potential failures
        if random.random() < 0.1: # 10% chance of failure
            return {"success": False, "error": "Simulated item out of stock"}
        else:
            mock_order_id = f"mock_{random.randint(10000, 99999)}"
            return {"success": True, "order_id": mock_order_id}

    def handle_intent(self, intent, entities, user_id, conversation_history):
        """Handles intents using extracted entities."""
        last_user_message = conversation_history[-1]['content'] if conversation_history else ""

        if intent == "place_order":
            order_items = entities.get("items", []) # Expecting a list like [{'name': 'X', 'quantity': Y}]
            if not order_items:
                print("SKILL: place_order intent received, but no items found in entities.")
                return "Okay, I can help with that. What items would you like to order?"


            # --- Use Mock API Call ---
            mock_response = self._make_mock_order_api_call(user_id, order_items)

            if mock_response["success"]:
                order_id = mock_response["order_id"]
                # Extract item names for confirmation message
                item_names = ", ".join([f"{item.get('quantity', 1)} {item.get('name', 'Unknown')}" for item in order_items])
                confirmation_message = f"Okay, your simulated order #{order_id} for {item_names} has been placed successfully!"
                print(f"MOCK API: Success - Order {order_id} placed for {item_names}.")
                # Update memory with actual order details if needed
                self.memory_handler.update_user_data(user_id, "last_order_items", item_names)
                self.memory_handler.update_user_data(user_id, "last_order_id", order_id)
            else:
                error_detail = mock_response["error"]
                confirmation_message = f"Sorry, there was a mock error placing your order: {error_detail}"
                print(f"MOCK API: Failure - {error_detail}")
            return confirmation_message
            # --- End Mock API ---

        elif intent in ["query_menu", "query_hours", "query_faq", "general_query"]:
            # For queries, just use the standard RAG + LLM response generated previously
            # The LLMHandler already produced the text response based on RAG context
            # We find the response in the conversation history
            # Find the last assistant message in the current history slice passed to the skill
            # This assumes the LLMHandler already generated the appropriate text response
            # based on RAG context for these query intents.
            last_assistant_message = next((msg['content'] for msg in reversed(conversation_history) if msg['role'] == 'assistant'), None)
            if last_assistant_message:
                 return last_assistant_message
            else:
                 # Fallback if somehow the response wasn't generated/found
                 return self.get_generic_response(last_user_message, user_id, conversation_history)


        else: # Unknown intent, treat as general query
            print(f"Warning: Skill received unhandled intent '{intent}'. Treating as general query.")
            # Fallback to generic RAG response
            # Re-run generic RAG if needed, or use the response already generated
            last_assistant_message = next((msg['content'] for msg in reversed(conversation_history) if msg['role'] == 'assistant'), None)
            return last_assistant_message or "Sorry, I'm not sure how to handle that specific request."