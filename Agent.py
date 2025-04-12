# Import required libraries
from langchain_ollama.llms import OllamaLLM  # For using the Ollama language model
from langchain_core.prompts import ChatPromptTemplate  # For creating chat prompts
from langchain_core.runnables.history import RunnableWithMessageHistory  # For conversation history management
from langchain_community.chat_message_histories import FileChatMessageHistory  # For persistent storage
from vector import retriver  # Import the retriever from vector.py
import os  # For file system operations

# Initialize the language model
# Using llama3.2 model from Ollama for generating responses
# This model will process the questions and generate answers
model = OllamaLLM(model="llama3.2")

# Create directory for storing chat history if it doesn't exist
# This directory will store JSON files containing conversation history
chat_history_dir = "./chat_history"
if not os.path.exists(chat_history_dir):
    os.makedirs(chat_history_dir)

# Define the template for the chat prompt
# This template will be used to format the input for the language model
# It includes placeholders for:
# - chat_history: Previous conversation context
# - reviews: Relevant reviews retrieved from the vector database
# - question: The user's current question
template = """
You are an expert in answering questions about Naruto verse. You have access to relevant reviews and previous conversation history.

Previous conversation:
{chat_history}

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# Create a prompt template from the template string
# This will be used to format the input for the language model
prompt = ChatPromptTemplate.from_template(template)

# Create a chain that combines the prompt and the model
# This chain will process the input and generate responses
chain = prompt | model

# Create a function to get chat history for a specific session
# This function is used by RunnableWithMessageHistory to manage conversation history
# Each session gets its own JSON file for storing chat history
def get_session_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(
        file_path=os.path.join(chat_history_dir, f"naruto_chat_history_{session_id}.json")
    )

# Create a chain with message history
# This combines the language model chain with conversation history management
# - input_messages_key: The key for the user's input in the prompt
# - history_messages_key: The key for the conversation history in the prompt
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Main interaction loop
while True:
    print("\n\n-----------------------------")
    # Get user input
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    
    # Check if user wants to quit
    if question == "q":
        break
        
    # Retrieve relevant reviews based on the question
    # This uses the retriever from vector.py to find similar reviews
    reviews = retriver.invoke(question)
    
    # Generate a response using the chain with history
    # The config parameter specifies which session to use
    # "default" is used as the session ID in this case
    result = chain_with_history.invoke(
        {
            "question": question,  # The user's question
            "reviews": reviews,    # Relevant reviews from the vector database
        },
        config={"configurable": {"session_id": "default"}},  # Use the default session
    )
    
    # Print the generated response
    print(result)