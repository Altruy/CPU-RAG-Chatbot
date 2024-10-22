# LLM-RAG Chatbot

This project is a CPU-based Retrieval-Augmented Generation (RAG) chatbot that leverages a Large Language Model (LLM) to answer user queries. The system uses LangChain for interaction and Chroma for vector database management. This chatbot is designed for document-based Question Answering (QA) by retrieving relevant context from documents and using that to generate accurate responses.

## Features

- **LLM-based Chatbot**: Uses a pre-trained model (LLaMA or Mistral) to answer queries with context-aware responses.
- **Document Retrieval**: Uses ChromaDB to store and retrieve relevant documents as context for answering questions.
- **Streamlit UI**: The chatbot is hosted on a web interface powered by Streamlit, with interactive chat functionality.
- **Contextual Querying**: Uses templates to provide context-aware responses based on retrieved documents.
  
## Requirements

- Python 3.9+
- Required Python Libraries:
  - `langchain-core`
  - `dotenv`
  - `streamlit`
  - `llama-cpp-python`
  - `chromadb`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/llm-rag-chatbot.git
    cd llm-rag-chatbot
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables by creating a `.env` file:
    ```bash
    touch .env
    ```

4. Place your model files in the `model-cache` directory. Ensure the model path in the code points to the correct location.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Upload your document (e.g., `change-management.pdf`) to be stored in ChromaDB for retrieval.

3. Interact with the chatbot through the Streamlit interface. You can ask any question, and the chatbot will use the provided context from the uploaded documents to generate a detailed response.

## Components

### 1. **LangChain Prompt Templates**
   The chatbot uses multiple prompt templates for different models. These prompts ensure that the model generates responses based on the context provided by the retrieved documents.

   - **Llama Template**: Designed for Meta's LLaMA model.
   - **Ollama Template**: For more general knowledge-based querying.
   - **Phi Template**: For use with the Phi model.

### 2. **Document Retrieval with ChromaDB**
   The documents uploaded by the user are indexed using ChromaDB. During interaction, the chatbot retrieves the most relevant documents and uses them as context to answer queries.

### 3. **Streamlit Interface**
   A simple chat UI that allows users to send messages and receive responses. The UI also manages session states to maintain a history of the conversation.

### 4. **Model Loading**
   The LLaMA model is loaded via `LlamaCpp`, ensuring efficient inference on CPU hardware.

## File Structure

```
llm-rag-chatbot/
├── model-cache/                # Directory for storing model files
├── utils_chromadb.py           # Utility functions for managing ChromaDB
├── app.py                      # Main Streamlit application
└── requirements.txt            # List of dependencies
```

## How It Works

1. **Model Initialization**: The LLaMA model is loaded from the `model-cache` directory.
2. **Document Upload & Indexing**: Users can upload documents (e.g., PDFs) which are processed and stored in ChromaDB for fast retrieval.
3. **Chat Interaction**: The user inputs a query through the Streamlit interface. The chatbot retrieves the most relevant document sections, generates a context-aware response, and sends it back to the user.
4. **Response Generation**: The chatbot uses the context retrieved from documents and a pre-defined prompt template to generate the response using the LLaMA model.

## Customization

- **Model**: You can replace the pre-trained model path with any other model you prefer, as long as it's supported by LlamaCpp.
- **Templates**: Modify the prompt templates in `app.py` to better fit your use case.
- **Documents**: Add your own documents by uploading them through the UI or manually integrating with ChromaDB.

## Future Enhancements

- Adding support for GPU-based inference.
- Improving document parsing for non-PDF formats.
- Integrating additional LLM models for better response quality.
  
## License

This project is licensed under the MIT License.

---

Feel free to contribute by opening pull requests or reporting issues!