import time
# Import updated handlers
from .audio_handler import AudioHandler
from .llm_handler import LLMHandler # Uses Llama.cpp now
from .rag_handler import RAGHandler
from ..memory.persistent_memory import PersistentMemory
from ..analysis.sentiment import SentimentAnalyzer
from ..skills.registry import get_skill_class

class ConversationManager:
    def __init__(self, config):
        self.config = config
        # Initialize updated handlers
        self.audio_handler = AudioHandler(config)
        self.llm_handler = LLMHandler(config) # Now uses Llama.cpp
        self.rag_handler = RAGHandler(config)
        self.memory_handler = PersistentMemory(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)

        # Load the active skill (same as before)
        active_skill_id = config.get('active_business_type', 'restaurant')
        SkillClass = get_skill_class(active_skill_id)
        self.active_skill = SkillClass(config, self.llm_handler, self.rag_handler, self.memory_handler)
        print(f"Active skill set: {self.active_skill.skill_id}")

        # Conversation state
        self.conversation_history = []
        self.max_history_pairs = 5
        self.current_user_id = "user_" + str(int(time.time()))
        self._initialize_history()

    # Inside src/core/conversation_manager.py

    def _initialize_history(self):
        """Sets up the initial system prompt asking for structured output."""
        # --- UPDATED SYSTEM PROMPT ---
        system_prompt = (
            f"You are a helpful AI assistant for the '{self.active_skill.skill_id}' business type."
            "Analyze the user's query, the provided 'Retrieved Context', and conversation history. "
            "Your goal is to identify the user's intent, extract key entities, and generate a concise, helpful response based ONLY on the provided information. "
            "Respond STRICTLY in the following JSON format ONLY:\n"
            "{\n"
            "  \"intent\": \"<intent_name>\",\n"
            "  \"entities\": { \"<key1>\": \"<value1>\", \"<key2>\": \"<value2>\", ... },\n"
            "  \"response\": \"<your_response_to_the_user>\"\n"
            "}\n\n"
            "Possible intents include: 'place_order', 'query_menu', 'query_hours', 'query_listings', 'schedule_viewing', 'query_faq', 'general_query', 'exit'.\n"
            "Extract relevant entities for the intent (e.g., for 'place_order', extract 'items' with 'name' and 'quantity'; for 'schedule_viewing', extract 'property_address' or 'property_id' and 'requested_time').\n"
            "Base the 'response' text ONLY on the retrieved context and history." 
            "If the information isn't available, state that in the response." 
            "Do not add apologies unless context suggests it. Do not use your general knowledge."
        )
        # -----------------------------
        user_data = self.memory_handler.get_user_data(self.current_user_id)
        user_name = user_data.get("name", None) # Example: trying to get name

        user_memory_summary = self.memory_handler.get_summary_for_prompt(self.current_user_id)
        initial_context = f"User Memory Summary:\n{user_memory_summary}"

        self.conversation_history = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'system', 'content': initial_context}
        ]
        print(f"Initialized conversation for user: {self.current_user_id}")
        print(f"User Memory: {user_memory_summary}")
        # Store name if found
        self.user_name = user_name

    def start_conversation(self):
        """Runs the main voice interaction loop with VAD."""
        # --- UPDATED GREETING ---
        greeting = "Hello! How can I help you today?"
        if self.user_name:
             greeting = f"Hello {self.user_name}! How can I assist you?"
        self.audio_handler.speak(greeting)

        while True:
            # --- 1. Record Audio using VAD ---
            # This now returns audio data directly, not a path
            audio_data_np = self.audio_handler.record_audio_with_vad()
            if audio_data_np is None:
                # No speech detected or too short, loop back silently to listen again
                time.sleep(0.1) # Small pause
                continue

            # --- 2. Transcribe Audio (using optimized whisper.cpp) ---
            user_input_text = self.audio_handler.transcribe_audio(audio_data_np)
            if user_input_text is None: # Transcription error
                self.audio_handler.speak("Transcription error. Please try again.")
                time.sleep(1); continue
            if not user_input_text or len(user_input_text.strip()) < 2: # Empty
                self.audio_handler.speak("Didn't catch that.")
                time.sleep(1); continue

            # Exit command check
            if any(word in user_input_text.lower() for word in ["quit", "exit", "goodbye"]):
                self.audio_handler.speak("Goodbye!"); break

            # --- 3. Analyze Sentiment ---
            sentiment = self.sentiment_analyzer.analyze(user_input_text)
            print(f"User Sentiment: {sentiment}")

            # --- 4. Retrieve RAG Context ---
            rag_context = self.rag_handler.retrieve_context(user_input_text)

            # --- 5. Prepare LLM Prompt ---
            self.conversation_history.append({'role': 'user', 'content': user_input_text})
            current_messages = []
            current_messages.append(self.conversation_history[0]) # System Prompt
            if rag_context and "Error" not in rag_context:
                current_messages.append({'role': 'system', 'content': f"Retrieved Context:\n{rag_context}"})
            elif "Error" in rag_context: print(f"Warning: RAG error: {rag_context}")
            # Add history (excluding system prompt)
            current_messages.extend(self.conversation_history[1:-1]) # Exclude latest user msg added above
            # Add latest user message again
            current_messages.append(self.conversation_history[-1])

            # --- 6. Get Structured Response (using optimized llama.cpp) ---
            llm_response_data = self.llm_handler.get_structured_response(current_messages)
            intent = llm_response_data.get("intent", "general_query")
            entities = llm_response_data.get("entities", {})
            # Get the INITIAL response text generated by the LLM based on RAG/History
            initial_ai_response_text = llm_response_data.get("response", "Sorry, I encountered an issue.")

            # --- 7. Delegate to Active Skill (if needed) ---
            # The skill handler might modify the response text (e.g., add confirmation)
            # or might just use the initial_ai_response_text if it's just a query.
            if intent != "general_query" and intent != "error":
                 # Skill handles specific actions and returns the FINAL text for these intents
                 final_ai_response_text = self.active_skill.handle_intent(
                     intent, entities, self.current_user_id, self.conversation_history
                 )
            else:
                 # For general queries or errors identified by LLM, use the text directly from LLM
                 final_ai_response_text = initial_ai_response_text


            # Update history with the FINAL AI response text (after potential skill modification)
            # Avoid adding duplicate if skill just returned the initial text? No, history needs the actual flow.
            self.conversation_history.append({'role': 'assistant', 'content': final_ai_response_text})

            # Limit history size
            if len(self.conversation_history) > (self.max_history_pairs * 2 + 2):
                 self.conversation_history = self.conversation_history[:2] + self.conversation_history[-(self.max_history_pairs * 2):]


            # --- 8. Optionally Modify Response Based on Sentiment ---
            # Apply modification to the FINAL text determined above
            final_spoken_response = final_ai_response_text
            # Example: Only add prefix if it wasn't an error and sentiment was negative
            if sentiment == "NEGATIVE" and intent != "error":
                import random # Make sure random is imported at the top of the file
                empathetic_phrases = ["I understand. ", "I'm sorry to hear that may be frustrating. ", "Okay, noting that. "]
                # Prepend the phrase
                final_spoken_response = random.choice(empathetic_phrases) + final_ai_response_text
                print(f"Adjusting response for negative sentiment.")


            # --- 9. Synthesize & Play Modified Response ---
            if not self.audio_handler.speak(final_spoken_response): # Use the potentially modified text
                 print("ERROR: Failed to speak the response.")
                 time.sleep(2); continue

            # --- 10. Cleanup Temp Audio Files ---
            self.audio_handler.cleanup_temp_files()
            print("-" * 20) # Separator
