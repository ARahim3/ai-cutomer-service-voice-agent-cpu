# AI Customer Service & Ordering Agent (Open-Source, Local)

## Introduction

This project implements an AI-powered customer service and ordering agent capable of interacting via voice, as outlined in the initial task description. The primary goal was to develop an adaptable, conversational agent using a **strictly open-source and locally executable technology stack**, balancing human-like interaction, dynamic knowledge retrieval, and performance constraints.

Given the project timeline and the complexity of certain requirements, this implementation focuses on delivering a robust core agent with modularity, optimized local AI models, and functional business logic using mocked backend integrations. Real-time telephony integration is designed conceptually but not fully implemented.

---

**Table of Contents**

*   [Features Implemented](#features-implemented)
*   [Setup & Installation](#setup--installation)
*   [Usage](#usage)
*   [Configuration](#configuration)
*   [Architecture Overview](#architecture-overview)
*   [Technology Stack](#technology-stack)
*   [Meeting Task Requirements](#meeting-task-requirements)
*   [Testing](#testing)
*   [Limitations](#limitations)
*   [Future Work](#future-work)
*   [Code working under the hood](#how-the-code-works-under-the-hood-processing-flow)
*   [Demo / Sample Interaction](#demo--sample-interaction)

---

## Features Implemented

*   **Voice Interaction:** Full voice-in, voice-out capability using local microphone and speakers.
*   **Voice Activity Detection (VAD):** Uses `webrtcvad` for automatic detection of speech start/end, enabling more natural turn-taking than fixed-duration recording.
*   **Optimized Local STT:** Employs the Hugging Face `transformers` pipeline (`openai/whisper-base.en`) potentially accelerated by ONNX Runtime for efficient Speech-to-Text.
*   **Optimized Local LLM:** Utilizes `llama-cpp-python` to run quantized GGUF language models (e.g., Qwen2.5 1.5B Instruct) locally for conversation logic, intent recognition, and response generation.
*   **Fast Local TTS:** Uses the efficient `kokoro` TTS engine (`hexgrad/Kokoro-82M`) for faster speech synthesis on CPU.
*   **Retrieval-Augmented Generation (RAG):** Implemented using LangChain, Sentence Transformers (`all-MiniLM-L6-v2`), and ChromaDB for dynamic retrieval of relevant information from a knowledge base.
*   **Adaptable Knowledge Base Indexing:** The provided `index_knowledge.py` script uses LangChain Document Loaders to process knowledge base files in various formats (JSON, PDF, TXT) from a designated directory, chunking and storing them in the ChromaDB vector store.
*   **Modular Code Architecture:** The codebase is structured into logical modules (`src/core`, `src/skills`, `src/memory`, `src/analysis`, `src/utils`) for better organization and maintainability.
*   **Multi-Business Adaptability:** Supports different business types (e.g., Restaurant, Real Estate) via configuration (`config.yaml`) and a Skill registry pattern. The active skill dictates the specific business logic applied.
*   **Business Skill Execution (Mocked):** Includes `RestaurantSkill` and `RealEstateSkill` that handle specific intents (e.g., `place_order`, `schedule_viewing`). **Backend API calls for these actions are currently mocked** within the skill handlers to demonstrate the complete functional flow.
*   **Persistent User Memory:** A simple file-based (`user_memory.json`) system remembers basic user details (e.g., last order, preferences) across sessions (based on a session-specific mock user ID).
*   **Basic Sentiment Analysis:** Uses a DistilBERT model to analyze user input sentiment (results are logged and minimally affect response tone).
*   **Configuration via YAML:** Centralized configuration (`src/config.yaml`) for models, paths, VAD parameters, active business type, etc.
*   **Basic Testing:** Includes representative unit and integration tests using `pytest` and `pytest-mock`.

## Setup & Installation

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/ARahim3/ai-cutomer-service-voice-agent-cpu.git
    cd ai-cutomer-service-voice-agent-cpu/
    ```
2. Download the `.gguf` model from [this link](https://huggingface.co/lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf?download=true) to the `models/` directory. 
3.  **Create & Activate Virtual Environment:**
    ```bash
    python -m venv venv # Or your preferred name like 'agent'
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
4.  **Install System Dependencies:**
    *   **Debian/Ubuntu:** `sudo apt update && sudo apt install espeak-ng build-essential` (build-essential might be needed for llama-cpp-python)
    *   **macOS:** `brew install espeak-ng` (ensure you have Xcode command line tools for potential compilation)
    *   **Windows:** Install `espeak-ng`. Installing C++ build tools (like Visual Studio Build Tools) might be needed for `llama-cpp-python`.
5.  **Install Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This might take time and require significant disk space).*
6.  **Download GGUF LLM Model:** Manual download required.
    *   Download a quantized GGUF model compatible with `llama-cpp-python` (e.g., `Qwen1.5-1.5B-Instruct-Q4_K_M.gguf`, `gemma-2b-it-q4_k_m.gguf`). Search Hugging Face for suitable models.
    *   Place the downloaded `.gguf` file inside the `models/` directory (create the directory if it doesn't exist).
7.  **Configure Model Paths:** Open `src/config.yaml` and ensure `llm.model_path` points to the correct filename of the GGUF model you downloaded inside the `models/` directory. Verify other settings like `stt.model_name` (`openai/whisper-base.en` is a good default).
8.  **Prepare Knowledge Base:**
    *   Place your knowledge base documents (e.g., `.json`, `.pdf`, `.txt`) inside the `knowledge_bases/` directory. Example files are included.
9.  **Run Knowledge Base Indexing:**
    ```bash
    python index_knowledge.py
    ```
    *   Run this once initially and again if KB files change. It creates the `chroma_db/` directory. The first run downloads embedding models.
10.  **(Optional) Environment Variables:** No environment variables are strictly required for the current mocked setup.

## Usage

1.  Ensure your virtual environment is activated.
2.  Ensure required models are downloaded (`models/`) and the knowledge base is indexed (`chroma_db/`).
3.  Run the agent from the **project root directory**:
    ```bash
    python -m src.main
    ```
4.  The agent will initialize (loading models might take time) and greet you.
5.  When you see "Listening...", speak your query clearly. Pause briefly when finished. VAD will detect the end of speech.
6.  The agent will process (Transcribe -> RAG -> LLM -> Skill -> TTS) and respond.
7.  To exit, say "quit", "exit", or "goodbye" when prompted to speak.

## Configuration

The agent's behavior is configured via `src/config.yaml`:

*   **`llm`**: Settings for the LLM handler (`provider`, `model_path` for GGUF, `n_ctx`, `n_gpu_layers`).
*   **`rag`**: Settings for RAG (`embedding_model`, `persist_directory` for ChromaDB, `collection_name_base`, `retrieval_k`).
*   **`stt`**: Settings for STT (`provider`, Hugging Face `model_name`).
*   **`tts`**: Settings for TTS (`provider`, specific Kokoro settings like `repo_id`, `lang_code`, `default_voice`).
*   **`audio`**: VAD parameters (`vad_aggressiveness`, `vad_max_silence_ms`, etc.) and sample rates.
*   **`memory`**: File path for persistent memory (`persist_file`).
*   **`sentiment`**: Hugging Face `model_name` for the sentiment analyzer.
*   **`active_business_type`**: Determines which skill module is loaded (must match a key in `src/skills/registry.py`, e.g., `restaurant`, `real_estate`).

To adapt to a new business (assuming KB format is compatible with indexer):
1.  Add KB files to `knowledge_bases/`.
2.  Run `python index_knowledge.py` (ensure `collection_name_base` covers the new data or use a different base and update config).
3.  Create a new Skill class in `src/skills/` (e.g., `src/skills/booking_skill.py`).
4.  Register the new skill in `src/skills/registry.py`.
5.  Update `active_business_type` in `src/config.yaml` to the new skill's ID.



## Architecture Overview

The agent operates based on the following core components and flow:

1.  **Audio Handler (`src/core/audio_handler.py`):** Manages audio input/output using `sounddevice`, `soundfile`, and `webrtcvad`.
2.  **STT Engine:** Hugging Face `transformers.pipeline` (`openai/whisper-base.en`).
3.  **TTS Engine:** `kokoro` (`hexgrad/Kokoro-82M`).
4.  **Conversation Manager (`src/core/conversation_manager.py`):** Orchestrates the main loop, manages state, calls handlers and skills.
5.  **RAG Handler (`src/core/rag_handler.py`):** Handles knowledge retrieval using LangChain, `HuggingFaceEmbeddings`, and `Chroma`.
6.  **LLM Handler (`src/core/llm_handler.py`):** Interfaces with the local LLM via `llama-cpp-python`, prompts for and parses structured JSON output (intent, entities, response).
7.  **Memory Handler (`src/memory/persistent_memory.py`):** Manages long-term user memory via a local JSON file.
8.  **Sentiment Analyzer (`src/analysis/sentiment.py`):** Uses a `transformers` pipeline for basic sentiment detection.
9.  **Skills (`src/skills/`):** Implement business-specific logic and **mocked API calls**. Loaded via `registry.py` based on config.
10. **Knowledge Base Indexer (`index_knowledge.py`):** Offline script using LangChain to process source documents (JSON, PDF, TXT) and populate the ChromaDB vector store.

**Conceptual Telephony Integration (Not Implemented):**
Live call handling would require integrating with Asterisk (via ARI) or FreeSWITCH (via ESL). The agent service would connect via WebSockets to receive/send audio streams. Streaming STT/TTS and further optimizations would be needed. This was deferred due to project constraints.

## Technology Stack

*   **Language:** Python 3.10+
*   **Core AI Models:**
    *   STT: Hugging Face Transformers Pipeline (`openai/whisper-base.en`) + ONNX Runtime
    *   LLM: Quantized GGUF Models (e.g., `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`) via `llama-cpp-python`.
    *   TTS: Kokoro (`hexgrad/Kokoro-82M`).
    *   Embeddings: Sentence Transformers (`all-MiniLM-L6-v2`) via `langchain-huggingface`.
    *   Sentiment: DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) via `transformers`.
*   **Frameworks/Libraries:**
    *   LangChain (`langchain`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`): For RAG pipeline.
    *   ChromaDB (`chromadb`): Local vector store.
    *   Audio: `sounddevice`, `soundfile`.
    *   VAD: `webrtcvad`.
    *   LLM Backend: `llama-cpp-python`.
    *   TTS Backend: `kokoro`.
    *   Configuration: `PyYAML`.
    *   Testing: `pytest`, `pytest-mock`.
    *   Dependencies: `numpy`, `torch`, `python-dotenv`, `jq`, `pypdf`, `transformers`, `optimum`, `onnxruntime`.
*   **System Dependencies:** `espeak-ng` (required by Kokoro TTS).


## Meeting Task Requirements

*   **Multi-Business Adaptability:** Achieved via configuration (`active_business_type`), skill registry, adaptable indexer (handling JSON/PDF/TXT), and modular design.
*   **Human-Like Conversational Ability:** Addressed through VAD, Kokoro TTS quality, conversational LLM, context handling, and basic sentiment analysis.
*   **Real-Time Call Handling:** **Not implemented.** A conceptual design involving Asterisk/ARI is outlined above. Significant optimizations (VAD, optimized STT/LLM/TTS backends) were performed on the local pipeline as a step towards this goal.
*   **Dynamic Knowledge Base Integration:** Fully implemented using LangChain RAG with ChromaDB vector store, supporting indexing from multiple file formats.
*   **Context Retention & Memory:** Implemented via conversation history tracking and persistent file-based memory (`PersistentMemory` module).
*   **Ordering & Inquiry Handling:** Implemented via the Skill architecture (`RestaurantSkill`, `RealEstateSkill`). **Backend API integration is MOCKED** to demonstrate the functional flow. Intents/Entities are extracted via the LLM (with current limitations based on model size).
*   **Open-Source Technology Stack:** Strictly adhered to using only open-source models (Whisper via Transformers/ONNX, LLM via Llama.cpp/GGUF, Kokoro TTS, Sentence Transformers, DistilBERT) and libraries (LangChain, ChromaDB, etc.) run locally. No proprietary cloud APIs are used.
*   **Performance & Optimization:** Addressed through VAD, use of optimized backends (`llama-cpp-python`, Kokoro TTS, Transformers/ONNX pipeline for STT), and quantized models (GGUF for LLM). **Latency limitations on CPU are acknowledged.**
*   **Security & Compliance:** Basic considerations noted (local execution, prompt design). Production systems would require enhanced security for data handling and API calls.
*   **Testing & Documentation:** Basic unit and integration tests provided using `pytest`. This README serves as comprehensive documentation.
*   **Deliverables:** Functional agent (local voice loop, RAG, skills with mocked APIs), modular codebase, configuration guide (this README), basic tests.

## Testing

This project uses `pytest`.

1.  Install test dependencies: `pip install pytest pytest-mock`
2.  From the project root directory, run:
    ```bash
    pytest
    ```
    or
    ```bash
    python -m pytest
    ```

```bash
$ pytest
============================= test session starts ==============================
platform linux -- Python 3.12.7, pytest-8.3.5, pluggy-1.5.0
rootdir: /home/rahim/agent_task
plugins: typeguard-4.4.2, langsmith-0.3.32, mock-3.14.0, anyio-4.9.0
collected 10 items                                                             

tests/integration/test_skills.py ....                                    [ 40%]
tests/unit/test_helpers.py ..                                            [ 60%]
tests/unit/test_memory.py ....                                           [100%]

============================== 10 passed in 0.15s ==============================
```
Tests cover configuration loading, memory persistence, and basic restaurant skill logic using mocked dependencies. See the `tests/` directory for details.

## Limitations

*   **No Live Telephony:** The agent runs locally and does not integrate with Asterisk/FreeSWITCH.
*   **Mocked API Calls:** Ordering and scheduling actions simulate backend calls but do not interact with real external APIs.
*   **Latency:** While optimized, noticeable latency may still occur during LLM inference and TTS synthesis, especially on lower-end CPUs. True real-time performance might require GPU acceleration or further streaming optimizations.
*   **NLU Accuracy:** Intent recognition and entity extraction rely on the chosen LLM's ability to follow complex prompting and generate structured JSON. Smaller models (like the ~1.5B parameter examples) may make errors or hallucinate entities. Reliability improves with larger models.
*   **VAD Sensitivity:** `webrtcvad` may require tuning (`vad_aggressiveness`, `vad_max_silence_ms` in config) depending on microphone quality and background noise.
*   **Error Handling:** Basic error handling is included, but a production system would require more comprehensive resilience.
*   **Sentiment Analysis Usage:** Sentiment is detected but only minimally used (optional response prefix); deeper integration is possible.

## Future Work

*   **Telephony Implementation:** Full integration with Asterisk/ARI for live call handling, including robust audio streaming.
*   **Replace Mocked APIs:** Implement real HTTP requests to backend services for ordering, scheduling, etc.
*   **Enhanced NLU:** Improve LLM prompting for more reliable JSON output (intent/entities), potentially explore function calling capabilities if supported by the local LLM runner.
*   **Advanced Streaming:** Implement true streaming TTS playback (playing audio chunks as they are generated).
*   **Testing:** Increase test coverage (more scenarios, edge cases, potentially E2E tests).
*   **Configuration:** Improve configuration management (e.g., mapping business types to specific KB collections).
*   **VAD:** Evaluate `pyannote.audio` for potentially more robust VAD if performance allows.
*   **Deployment:** Enhance Dockerfile, potentially add `docker-compose` for easier multi-container setups (e.g., separate Qdrant server if switching from Chroma).


## How the code Works: Under the Hood (Processing Flow)

When you speak to the agent after seeing the "Listening..." message, here's a simplified step-by-step breakdown of what happens internally:

1.  **Listening & VAD (Audio Handler):** The system uses the microphone (`sounddevice`) and Voice Activity Detection (`webrtcvad`) to listen continuously. It buffers small audio chunks.
2.  **Speech Detection (Audio Handler):** When VAD detects speech significantly longer than noise/silence, it starts saving the audio chunks.
3.  **Silence Detection (Audio Handler):** When VAD detects a sufficient pause after speech, it stops saving audio.
4.  **Transcription (Audio Handler -> STT Engine):** The collected raw audio data (as a NumPy array) is passed to the optimized STT engine (Hugging Face pipeline using `openai/whisper-base.en` with potential ONNX acceleration).
5.  **Text Output (STT Engine -> Conversation Manager):** The STT engine returns the transcribed text of what you said.
6.  **Sentiment Analysis (Conversation Manager -> Sentiment Analyzer):** The transcribed text is briefly analyzed (using DistilBERT) to gauge user sentiment (Positive/Negative/Neutral), which is logged.
7.  **Context Retrieval (Conversation Manager -> RAG Handler):** The transcribed text (your query) is sent to the RAG Handler.
    *   **Embedding:** The query is converted into a numerical vector (embedding) using the Sentence Transformer model (`all-MiniLM-L6-v2`).
    *   **Vector Search:** The ChromaDB vector store is searched to find the top `k` (e.g., 3) text chunks from the indexed Knowledge Base that are semantically most similar to the query embedding.
    *   **Context Formatting:** The text content of these retrieved chunks is formatted into a single string.
8.  **LLM Prompting (Conversation Manager -> LLM Handler):**
    *   The Conversation Manager assembles a prompt for the local LLM (`llama-cpp-python` running a GGUF model like Qwen).
    *   This prompt includes:
        *   A system instruction (defining the agent's role, how to use context, and requesting JSON output).
        *   A summary of any relevant user memory (from `user_memory.json`).
        *   The retrieved RAG context chunks.
        *   The recent conversation history.
        *   Your latest transcribed query.
9.  **LLM Processing (LLM Handler -> Llama.cpp):** The LLM processes the complex prompt to:
    *   Understand the query in context.
    *   Identify the user's intent (e.g., `place_order`, `query_menu`).
    *   Extract relevant entities (e.g., item names, property details - *currently basic*).
    *   Generate a conversational response based *only* on the provided context/history.
    *   Format its entire output as a JSON object containing `intent`, `entities`, and `response`.
10. **Response Parsing (LLM Handler -> Conversation Manager):** The LLM Handler receives the raw text, attempts to parse the JSON, and extracts the `intent`, `entities`, and `response` text. (Includes fallback if JSON is invalid).
11. **Skill Delegation (Conversation Manager -> Active Skill):**
    *   If the identified `intent` requires a specific action (like `place_order`), the `ConversationManager` calls the `handle_intent` method of the currently active skill (e.g., `RestaurantSkill`).
    *   The skill uses the extracted `entities`, performs its logic (including **mocked API calls**), potentially updates user memory, and returns the final text response for the user (e.g., an order confirmation).
    *   If the intent is a general query, the response text generated directly by the LLM (in Step 10) is typically used.
12. **(Optional) Sentiment Adjustment (Conversation Manager):** A simple prefix might be added to the final response text based on the earlier sentiment analysis (e.g., adding "Okay." if sentiment was negative).
13. **Speech Synthesis (Conversation Manager -> Audio Handler -> TTS Engine):** The final text response is sent to the `AudioHandler`.
    *   The handler calls the Kokoro TTS engine.
    *   Kokoro generates the audio data chunk by chunk.
    *   The `AudioHandler` concatenates these chunks and saves the complete audio to a temporary `.wav` file.
14. **Playback (Audio Handler):** The `AudioHandler` immediately plays the generated `.wav` file through your speakers using `sounddevice` and `soundfile`.
15. **Cleanup (Audio Handler):** Temporary audio files are deleted.
16. **Loop:** The agent returns to the "Listening..." state (Step 1).

This flow highlights how different modules collaborate, using optimized local models and RAG to process your voice request and generate a relevant spoken response.


## Demo / Sample Interaction (Illustrative)

```
(Agent starts)
AI Speaking: Hello! How can I help you today?
Listening... (Speak clearly, pause to finish)
(User speaks: "Tell me about the pixel pasta")
Speech detected, recording...
Silence detected, stopping recording.
Recording finished (X.XXs).
Transcribing audio (insanely-fast-whisper)...
Transcription: Tell me about the pixel pasta
User Sentiment: NEUTRAL
Retrieving context for query: 'Tell me about the pixel pasta'
Retrieved 3 context chunks.
Sending query to LLM (llama.cpp). Expecting JSON output...
LLM Raw Output:
{
  "intent": "query_menu",
  "entities": {"item_name": "Pixel Pasta"},
  "response": "Pixel Pasta is Tagliatelle pasta with a vibrant, pixel-art inspired vegetable medley and a light garlic sauce. It costs $12.99 and a vegetarian option is available."
}
Parsed Intent: query_menu, Entities: {'item_name': 'Pixel Pasta'}
AI Speaking: Pixel Pasta is Tagliatelle pasta with a vibrant, pixel-art inspired vegetable medley and a light garlic sauce. It costs $12.99 and a vegetarian option is available.
--------------------
Listening... (Speak clearly, pause to finish)
(User speaks: "I want to order one")
Speech detected, recording...
Silence detected, stopping recording.
Recording finished (Y.YYs).
Transcribing audio (insanely-fast-whisper)...
Transcription: I want to order one
User Sentiment: POSITIVE
Retrieving context for query: 'I want to order one'
Retrieved 3 context chunks.
Sending query to LLM (llama.cpp). Expecting JSON output...
LLM Raw Output:
{
  "intent": "place_order",
  "entities": { "items": [{"name": "Pixel Pasta", "quantity": 1}]},
  "response": "Okay, adding one Pixel Pasta to your order."
}
Parsed Intent: place_order, Entities: {'items': [{'name': 'Pixel Pasta', 'quantity': 1}]}
SKILL: Received intent 'place_order' with entities: {'items': [{'name': 'Pixel Pasta', 'quantity': 1}]}
MOCK API CALL: Placing order for user user_XXXXXXXXXX, items: [{'name': 'Pixel Pasta', 'quantity': 1}]
MOCK API: Success - Order mock_12345 placed for 1 Pixel Pasta.
AI Speaking: Okay, your simulated order #mock_12345 for 1 Pixel Pasta has been placed successfully!
--------------------
Listening... (Speak clearly, pause to finish)
```