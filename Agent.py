# Import required libraries
from langchain_ollama.llms import OllamaLLM  # For using the Ollama language model
from langchain_core.prompts import ChatPromptTemplate  # For creating chat prompts
from vector import retriver  # Import the retriever from vector.py

# Initialize the language model
# Using llama3.2 model from Ollama
model = OllamaLLM(model="llama3.2")

# Define the template for the chat prompt
# This template will be used to format the input for the language model
template = """
You are an expert in answering questions about Naruto verse

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# Create a prompt template from the template string
prompt = ChatPromptTemplate.from_template(template)

# Create a chain that combines the prompt and the model
# This chain will process the input and generate responses
chain = prompt | model

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
    reviews = retriver.invoke(question)
    
    # Generate a response using the chain
    # The chain combines the prompt template and the language model
    result = chain.invoke({
        "reviews": reviews,  # Pass the retrieved reviews
        "question": question  # Pass the user's question
    })
    
    # Print the generated response
    print(result)