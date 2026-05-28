import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DATA_PATH = "data"
DB_PATH = "vectorstore"


def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)

        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Vector database created successfully.")


def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Creating vector database...")
    create_vectorstore(chunks)


if __name__ == "__main__":
    main()