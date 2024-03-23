# Document Question Answering using LangChain, F.A.I.S.S. and Gemma

## Overview
This project allows you to ask questions from the document (PDF). This project utilizes the Retrieval Augmented Generation (RAG) concept employing a Retriever-Generator approach. The Retriever comprises an embedding model and vector database tasked with retrieving relevant information or context based on user queries. On the other hand, the Generator, implemented as an LLM with customized prompts, generates answers based on the retrieved information or context.

Initial project screen:
![Screenshot 2024-03-23 at 16 50 13](https://github.com/crypter70/Document-Question-Answering-using-LangChain-F.A.I.S.S.-and-Gemma/assets/74947224/5625e9ce-a649-4d09-9f7b-9de2d4c118a7)
Project screen after results:
![Screenshot 2024-03-23 at 16 52 55](https://github.com/crypter70/Document-Question-Answering-using-LangChain-F.A.I.S.S.-and-Gemma/assets/74947224/a02d371b-e307-4c32-b3a4-5d30ad0d1c38)

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
