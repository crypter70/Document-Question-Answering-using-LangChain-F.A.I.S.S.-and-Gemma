# Document Question Answering using LangChain, F.A.I.S.S. and Gemma

## Overview
This project allows you to ask questions from the document (PDF). This project utilizes the Retrieval Augmented Generation (RAG) concept employing a Retriever-Generator approach. The Retriever comprises an embedding model and vector database tasked with retrieving relevant information or context based on user queries. On the other hand, the Generator, implemented as an LLM with customized prompts, generates answers based on the retrieved information or context.

Intial project screen:

Project screen after results:

## Installation
1. Clone the repository
    ```
    git clone https://github.com/crypter70/Document-Question-Answering-using-LangChain-F.A.I.S.S.-and-Gemma
    ```

2. Create a virtual environment
    ```
    python -m venv env
    ```

3. Install the requirements
    ```
    pip install -r requirements.txt
    ```

4. Add the HUGGINGFACEHUB_API_TOKEN from [HuggingFace](https://huggingface.co/settings/tokens)
 to .env file

5. Run the application
    ```
    streamlit run App.py
    ```

## Reference
1. [LangChain Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
2. [gemma-7b-it](https://huggingface.co/google/gemma-7b-it)
