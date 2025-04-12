# Naruto Q&A Agent

A specialized question-answering agent for the Naruto anime/manga series, built using LangChain, Ollama, and ChromaDB. This agent uses advanced natural language processing to provide accurate and contextually relevant answers about the Naruto universe.

## Features

- **Persistent Memory**: Remembers previous conversations and maintains context across sessions
- **Semantic Search**: Finds relevant reviews using advanced vector embeddings
- **Natural Language Understanding**: Uses state-of-the-art language models for accurate responses
- **File-based Storage**: Saves chat history in JSON format for persistence

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/swoyamsss/Naruto-qa-agent.git
   cd Naruto-qa-agent
   ```

2. Create and activate a virtual environment:

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install langchain-ollama langchain-chroma langchain-core langchain-community pandas
   ```

4. Install Ollama and required models:
   ```bash
   # Install Ollama from https://ollama.ai/
   # Then install the required models:
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

## Components

### 1. Data Storage (`vector.py`)

- **CSV Data**: Contains Naruto reviews with titles, dates, character information, ratings, and reviews
- **Vector Database**: Uses ChromaDB to store and retrieve document embeddings
- **Embedding Model**: Uses Ollama's mxbai-embed-large model to generate embeddings
- **Retriever**: Implements semantic search to find relevant reviews based on user queries

### 2. Question Answering System (`Agent.py`)

- **Language Model**: Uses Ollama's llama3.2 model for generating responses
- **Prompt Template**: Structured template for formatting questions and context
- **Memory System**:
  - Uses `RunnableWithMessageHistory` for conversation management
  - Stores chat history in JSON files per session
  - Maintains context across multiple interactions
- **Retrieval System**: Finds relevant reviews based on user questions

## How It Works

1. **Initialization**:

   - The system loads the Naruto reviews from the CSV file
   - Creates vector embeddings for each review using Ollama's embedding model
   - Stores these embeddings in a ChromaDB vector store
   - Sets up the chat history storage system

2. **User Interaction**:

   - User asks a question about Naruto
   - The system retrieves the 10 most relevant reviews using semantic search
   - Loads previous conversation history from the session file
   - Formats the prompt with the question, relevant reviews, and conversation history
   - Generates a response using the language model
   - Saves the conversation to the session history file

3. **Memory Management**:
   - Each session's chat history is stored in a separate JSON file
   - The system automatically loads previous conversations when restarting
   - Context is maintained across multiple questions in the same session
   - History files are stored in the `./chat_history` directory

## Flow Diagram

```mermaid
graph TD
    A[User Question] --> B[Retrieve Relevant Reviews]
    B --> C[Load Conversation History]
    C --> D[Format Prompt with Reviews, History & Question]
    D --> E[Generate Response with LLM]
    E --> F[Save to Chat History]
    F --> G[Return Answer to User]

    H[CSV Data] --> I[Create Embeddings]
    I --> J[Store in Vector DB]
    J --> B

    K[Chat History Files] --> C
    F --> K
```

## Usage

1. Run the agent:

   ```bash
   python Agent.py
   ```

2. Ask questions about Naruto:
   - Type your question and press Enter
   - The agent will remember previous conversations
   - Type 'q' to quit

## File Structure

- `Agent.py`: Main agent implementation with memory and conversation handling
- `vector.py`: Vector database and retrieval system
- `Naruto_reviews.csv`: Dataset containing Naruto reviews
- `./chat_history/`: Directory containing session chat history files
- `./chroma_langchain_db/`: Directory containing the vector database

## Dependencies

- langchain-ollama
- langchain-chroma
- langchain-core
- langchain-community
- pandas
- ollama (for running the language models)

## Note

Make sure you have Ollama installed and running with the required models (llama3.2 and mxbai-embed-large) before using this agent.

## Author

- [Swoyam Sambhab Sahoo](https://github.com/swoyamsss)
