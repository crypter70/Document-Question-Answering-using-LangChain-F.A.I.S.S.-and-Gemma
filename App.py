

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

repo_id = "google/gemma-7b-it"
llm = HuggingFaceEndpoint(
    repo_id=repo_id
)

chara_text_splitter = CharacterTextSplitter()
recur_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)


def preprocessing_text(data):
    texts = chara_text_splitter.split_text(data)
    doc_data = [Document(page_content=t) for t in texts]
    text_splits = recur_text_splitter.split_documents(doc_data)
    return text_splits


def create_retriever(text_splits):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.from_documents(text_splits, embedding_model)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever    


def generate_answer(llm, retriever, query):
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    print(format_docs)

    return qa_chain.invoke(query)


def main():
    
    st.title("Document Question Answering 📝")
    st.subheader("Powered by LangChain, F.A.I.S.S., and Gemma")

    data = st.file_uploader('Upload your PDF document', type="pdf")
    if data is not None:
        pdf_reader = PdfReader(data)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        preprocessed_text = preprocessing_text(text)
        retriever = create_retriever(preprocessed_text)

    query = st.text_area('Ask a question:')

    if st.button("Ask"):
        if not data and not query:
            st.error("Please upload your PDF document or fill your question!")
        elif not data:
            st.error("Please upload your PDF document!")
        elif not query:
            st.error("Please fill your question!")
        else:
            with st.spinner('Answering question...'):
                answer = generate_answer(llm, retriever, query)
            st.subheader("Answer:")
            st.write(answer) 
            

if __name__ == "__main__":
    main()
