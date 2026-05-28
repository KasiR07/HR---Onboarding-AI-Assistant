import streamlit as st

# Vector database
from langchain_community.vectorstores import Chroma

# Embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Local LLM through Ollama
from langchain_ollama import OllamaLLM

# Prompt template utilities
from langchain_core.prompts import ChatPromptTemplate

# Helps pass the user question into the chain
from langchain_core.runnables import RunnablePassthrough


# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------

# Folder where ChromaDB stores vector embeddings
DB_PATH = "vectorstore"


# ---------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ---------------------------------------------------

st.set_page_config(
    page_title="Onboarding Copilot",
    page_icon="🤖",
    layout="centered"
)


# ---------------------------------------------------
# APP TITLE
# ---------------------------------------------------

st.title("🤖 Onboarding Copilot")

st.markdown("""
This assistant helps new employees quickly find information from:

- HR policies
- IT SOPs
- Onboarding guides
- Internal procedures

The chatbot uses Retrieval-Augmented Generation (RAG)
to retrieve relevant company information before generating answers.
""")


# ---------------------------------------------------
# LOAD VECTOR DATABASE
# ---------------------------------------------------

@st.cache_resource
def load_vector_database():
    """
    Loads the Chroma vector database.

    Uses HuggingFace embeddings to convert
    text into vector representations.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    return vector_db


# ---------------------------------------------------
# LOAD LOCAL OLLAMA MODEL
# ---------------------------------------------------

@st.cache_resource
def load_llm():
    """
    Loads the local Ollama LLM.

    Make sure:
    ollama run llama3.1

    is already running in another terminal.
    """

    llm = OllamaLLM(
        model="llama3.1"
    )

    return llm


# ---------------------------------------------------
# FORMAT RETRIEVED DOCUMENTS
# ---------------------------------------------------

def format_retrieved_docs(documents):
    """
    Combines retrieved document chunks
    into a single context string.
    """

    return "\n\n".join(
        doc.page_content for doc in documents
    )


# ---------------------------------------------------
# INITIALIZE COMPONENTS
# ---------------------------------------------------

# Load vector database
vectorstore = load_vector_database()

# Retriever searches top matching chunks
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# Load local language model
llm = load_llm()


# ---------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------

prompt_template = """
You are an onboarding assistant for employees.

Use ONLY the provided company documents to answer the user's question. Make sure not to guess or hallucinate any info.

If the answer is not present in the documents,
respond with:

"I could not find this information in the onboarding documents."

Keep all the responses professional, concise, and easy to understand.

Context:
{context}

Question:
{question}

Answer:
"""


prompt = ChatPromptTemplate.from_template(
    prompt_template
)


# ---------------------------------------------------
# CREATE RAG PIPELINE
# ---------------------------------------------------

rag_chain = (
    {
        # Retrieve relevant document chunks
        "context": retriever | format_retrieved_docs,

        # Pass user question directly
        "question": RunnablePassthrough()
    }

    # Insert values into prompt template
    | prompt

    # Generate final response using Ollama
    | llm
)


# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

user_question = st.text_input(
    "Ask a question about company onboarding:"
)


# ---------------------------------------------------
# HANDLE QUESTION
# ---------------------------------------------------

if user_question:

    with st.spinner("Searching company documents..."):

        # Generate answer
        response = rag_chain.invoke(user_question)

        # Retrieve relevant documents separately
        retrieved_docs = retriever.invoke(user_question)

    # ---------------------------------------------------
    # DISPLAY RESPONSE
    # ---------------------------------------------------

    st.subheader("Answer")

    st.write(response)

    # ---------------------------------------------------
    # DISPLAY RETRIEVED SOURCES
    # ---------------------------------------------------

    st.subheader("Retrieved Sources")

    for index, doc in enumerate(retrieved_docs, start=1):

        source_name = doc.metadata.get(
            "source",
            "Unknown source"
        )

        st.markdown(f"### Source {index}")

        st.write(f"Document: {source_name}")

        st.caption(
            doc.page_content[:400] + "..."
        )