from llama_cpp import Llama
import json
import os

class LLMHandler:
    def __init__(self, config):
        self.config = config['llm']
        self.model_path = self.config['model_path']
        self.n_ctx = self.config.get('n_ctx', 2048)
        self.n_gpu_layers = self.config.get('n_gpu_layers', 0) # Default to CPU

        print(f"Initializing LLM Handler ({self.config['provider']}: {self.model_path})...")
        self._load_model()
        print("LLM Handler initialized.")

    def _load_model(self):
        if self.config['provider'] == 'llama.cpp':
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"LLM GGUF model not found at: {self.model_path}")
            try:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False # Set to True for more debug info
                )
                print("Llama.cpp model loaded.")
            except Exception as e:
                print(f"Error loading Llama.cpp model: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['provider']}")

    def get_structured_response(self, messages):
        """Sends messages to the LLM and parses the expected JSON response."""
        print(f"Sending query to LLM ({self.config['provider']}). Expecting JSON output...")

        if self.config['provider'] == 'llama.cpp':
            try:
                # Send messages with instructions already in system prompt
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=512, # Adjust if needed for longer JSON + response
                    temperature=0.5, # Lower temperature might help consistency for JSON
                    # Consider adding specific JSON format hints if needed, e.g. via stop tokens
                    # stop=["}"] # Might stop generation too early sometimes
                )
                raw_response_text = response['choices'][0]['message']['content'].strip()
                print(f"LLM Raw Output:\n{raw_response_text}") # Log the raw output for debugging

                # --- Attempt to parse JSON ---
                try:
                    # Find the start and end of the JSON block (simple parsing)
                    json_start = raw_response_text.find('{')
                    json_end = raw_response_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_string = raw_response_text[json_start:json_end]
                        parsed_data = json.loads(json_string)

                        # Validate expected keys
                        intent = parsed_data.get("intent", "general_query")
                        entities = parsed_data.get("entities", {})
                        ai_response_text = parsed_data.get("response", "Sorry, I couldn't formulate a response.")

                        print(f"Parsed Intent: {intent}, Entities: {entities}")
                        return {"response": ai_response_text, "intent": intent, "entities": entities}
                    else:
                        print("Warning: Could not find JSON object in LLM response.")
                        # Fallback: Use the raw text as response, default intent
                        return {"response": raw_response_text, "intent": "general_query", "entities": {}}

                except json.JSONDecodeError as json_e:
                    print(f"Warning: LLM response was not valid JSON: {json_e}")
                    print(f"LLM Raw Output was: {raw_response_text}")
                    # Return a user-friendly error message, not the raw broken JSON
                    fallback_response = "I seem to have had trouble formatting my response correctly. Could you please try rephrasing?"
                    return {"response": fallback_response, "intent": "error", "entities": {}}
                # --- End JSON Parsing ---

            except Exception as e:
                print(f"Error calling Llama.cpp model: {e}")
                return {"response": "Error processing request.", "intent": "error", "entities": {}}
        else:
            # ... (handle other providers if any) ...
            return {"response": "Unsupported LLM provider.", "intent": "error", "entities": {}}