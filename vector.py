# Import required libraries
from langchain_ollama import OllamaEmbeddings  # For generating embeddings using Ollama
from langchain_chroma import Chroma  # For vector storage and retrieval
from langchain_core.documents import Document  # For creating document objects
import os  # For file system operations
import pandas as pd  # For CSV data handling

# Load the Naruto reviews dataset
# The CSV file contains reviews with columns: Title, Date, Best Character, Rating, Review
df = pd.read_csv("Naruto_reviews.csv")

# Initialize the embedding model
# Using mxbai-embed-large model from Ollama for generating embeddings
# This model converts text into high-dimensional vectors for semantic search
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define the location for the vector database
# This directory will store the ChromaDB database files
db_location = "./chroma_langchain_db"

# Check if we need to create new documents or use existing ones
# If the database doesn't exist, we'll create it
add_documents = not os.path.exists(db_location)

# If we need to create new documents
if add_documents:
    # Initialize lists to store documents and their IDs
    documents = []
    ids = []

    # Process each row in the dataframe
    for i, row in df.iterrows():
        # Create a Document object for each review
        # Combine Title and Review for the main content
        # Store Rating, Date, and Best Character as metadata
        document = Document(
            page_content=row["Title"] + " " + row["Review"],  # Combine title and review for better context
            metadata = {
                "rating": row["Rating"],  # Store rating as metadata
                "date": row["Date"],  # Store date as metadata
                "best character": row["Best Character"]  # Store best character as metadata
            },
            id = str(i)  # Use row index as document ID
        )
        ids.append(str(i))
        documents.append(document)

# Initialize the Chroma vector store
# This will either create a new database or load an existing one
vector_store = Chroma(
    collection_name="Naruto_Verse",  # Name of the collection
    persist_directory=db_location,    # Where to store/load the database
    embedding_function=embeddings     # Which embedding model to use
)

# If we have new documents, add them to the vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create a retriever from the vector store
# This will be used to find relevant documents based on queries
# k=10 means it will return the 10 most relevant documents
retriver = vector_store.as_retriever(
    search_kwargs={"k":10}  # Number of most relevant documents to retrieve
)