import os
import pickle
import streamlit as st
import time

from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

filepath = "vectorstore.pkl"
llm = OpenAI(temperature=0.9, max_tokens=500)

st.title("News Research Tool")

st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if len(url)> 0:
        urls.append(url)

click = st.sidebar.button("Process")
main_placeholder = st.empty()

if click:
    if len(urls) == 0:
        st.warning("Please input at least one URL")
        st.stop()
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data...")
    data = loader.load()
    # Split data
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=0
    )
    main_placeholder.text("Splitting data...")
    docs = splitter.split_documents(data)
    # Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Creating embeddings...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    # Save the FAISS index to a pickle file
    with open(filepath, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, vectorstore=vectorstore.as_retriever())
        main_placeholder.text("Searching...")
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result.get("answer"))

        # Display sources
        sources = result.get("sources", [])
        if sources:
            st.subheader("Sources:")
            source_list = sources.split("\n")
            for source in source_list:
                st.write(source)