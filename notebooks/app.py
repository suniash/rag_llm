import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import os
import openai

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
openai.api_key = os.environ['OPENAI_API_KEY']

# Step 1: Load the FAISS index
@st.cache_resource
def load_faiss_index():
    try:
        # Adjust path to your FAISS index
        faiss_index = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    except RuntimeError as e:
        st.error(f"Failed to load FAISS index: {str(e)}")
        return None
    return faiss_index

# Load the FAISS index
faiss_index = load_faiss_index()

if faiss_index is not None:
    # Step 2: Set up the Streamlit UI
    st.title("RAG-based Chatbot")

    # User input for the query
    user_query = st.text_input("Ask me anything about the document:")

    # Step 3: Handle the query and generate a response
    if user_query:
        # Perform a similarity search to find the most relevant chunks
        retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Set up the conversational retrieval chain without memory
        llm = OpenAI()
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=None)

        # Query the stored embeddings for similar documents
        response = qa_chain.run({"question": user_query, "chat_history": []})

        # Display the response
        st.write("Response:")
        st.write(response)
else:
    st.write("FAISS index could not be loaded. Please check the path and file.")
