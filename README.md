# Onboarding Copilot for New Hires

An enterprise-style Retrieval-Augmented Generation (RAG) chatbot built using LangChain, HuggingFace embeddings, ChromaDB, Ollama, and Streamlit.

This project helps new employees retrieve accurate answers from HR policies, onboarding guides, and IT SOPs using semantic search and local LLM inference.

---

# Features

* Semantic document retrieval
* Local LLM inference using Ollama
* ChromaDB vector database
* HuggingFace sentence embeddings
* Streamlit chatbot interface
* Source-aware responses
* HR onboarding knowledge assistant

---

# Tech Stack

* Python
* LangChain
* HuggingFace
* ChromaDB
* Ollama
* Streamlit

---

# Architecture

```text
Company Documents
       ↓
Document Chunking
       ↓
HuggingFace Embeddings
       ↓
ChromaDB Vector Store
       ↓
Semantic Retrieval
       ↓
Ollama LLM
       ↓
Streamlit Chat Interface
```

---

# Project Structure

```text
onboarding-assistant/
│
├── app.py
├── ingest.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── hr_policy.txt
│   ├── onboarding_guide.txt
│   └── it_sop.txt
```

---

# Setup Instructions

## Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Generate embeddings and vector database

```bash
python ingest.py
```

## Run Ollama

```bash
ollama run llama3.1
```

## Start Streamlit app

```bash
streamlit run app.py
```

---

# Example Questions

* How many vacation days do employees receive?
* What tools are used for collaboration?
* How does VPN approval work?
* What mandatory training must employees complete?
* Who handles payroll concerns?

---

# Future Improvements

* Slack integration
* PDF upload support
* Hybrid retrieval
* Authentication system
* Azure AI Search integration
* Conversation memory
* Admin dashboard
