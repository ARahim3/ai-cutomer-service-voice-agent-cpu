from abc import ABC, abstractmethod

class BaseSkill(ABC):
    """Abstract base class for business-specific skills."""
    skill_id = "base" # Unique identifier for the skill

    def __init__(self, config, llm_handler, rag_handler, memory_handler):
        self.config = config # Global config
        self.llm_handler = llm_handler
        self.rag_handler = rag_handler
        self.memory_handler = memory_handler
        print(f"Initializing skill: {self.skill_id}")

    @abstractmethod
    def handle_intent(self, intent, entities, user_id, conversation_history):
        """
        Process the recognized intent and entities.
        Should return the text response for the user.
        """
        pass

    def get_generic_response(self, prompt, user_id, conversation_history):
        """Handles general queries using RAG and LLM."""
        rag_context = self.rag_handler.retrieve_context(prompt)
        memory_summary = self.memory_handler.get_summary_for_prompt(user_id)

        messages = conversation_history.copy()
        # Add context and memory to the start or as system messages
        context_prompt = (
            f"Retrieved Context:\n{rag_context}\n\n"
            f"User Memory Summary:\n{memory_summary}"
        )
        messages.insert(1, {'role': 'system', 'content': context_prompt}) # Insert after initial system prompt
        messages.append({'role': 'user', 'content': prompt})

        llm_response_data = self.llm_handler.get_structured_response(messages)
        # For generic responses, we mainly care about the text
        return llm_response_data.get("response", "Sorry, I couldn't process that.")