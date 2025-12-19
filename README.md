# 🧠 Local Chat

**DocuChat Local** is a privacy-first RAG (Retrieval-Augmented Generation) application that allows you to chat with your PDF documents entirely offline. Built with Streamlit, LangGraph, and Ollama.

## Key Features

* **100% Local Privacy:** No data leaves your machine. Uses local LLMs via Ollama.
* **Smart Caching:** Implements SHA-256 hashing to cache processed vectors. Re-uploading the same files takes seconds, not minutes.
* **Robust Architecture:** built on **LangGraph** for reliable state management (Retrieve → Generate flow).
* **Source Citations:** Every answer cites the specific source file used for context.
* **Configurable:** Adjust chunk sizes, retrieval `k`, and switch models (Llama 3, Mistral, etc.) directly from the UI.

## Tech Stack

* **UI:** Streamlit
* **Orchestration:** LangChain & LangGraph
* **Vector Store:** FAISS
* **LLM Engine:** Ollama

## Prerequisites

1.  **Python 3.10+**
2.  **Ollama** installed and running.

### Required Models
You must pull the models used in the default configuration:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
