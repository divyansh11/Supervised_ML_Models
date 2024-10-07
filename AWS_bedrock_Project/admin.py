import boto3
import boto3.s3
import streamlit as st
import os
import uuid

##S3 Client
s3_client=boto3.client("s3")
# bucket_name=os.getenv("BUCKET_NAME")

bucket_name="divyansh-bedrock"
## Langchain Embeddings
from langchain_community.embeddings import BedrockEmbeddings

## Langchain Textsplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf loader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime",region_name="ap-south-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())


## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=bucket_name, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=bucket_name, Key="my_faiss.pkl")

    return True

uploaded_file=st.file_uploader('Choose a file to upload',"pdf")

if uploaded_file is not None:
    request_id = get_unique_id()
    st.write(f"Request Id: {request_id}")
    saved_file_name = f"{request_id}.pdf"
    with open(saved_file_name, mode="wb") as w:
        w.write(uploaded_file.getvalue())

    loader = PyPDFLoader(saved_file_name)
    pages = loader.load_and_split()

    st.write(f"Total Pages: {len(pages)}")

    ## Split Text
    splitted_docs = split_text(pages, 1000, 200)
    st.write(f"Splitted Docs length: {len(splitted_docs)}")
    st.write("===================")
    st.write(splitted_docs[0])
    st.write("===================")
    st.write(splitted_docs[1])

    st.write("Creating the Vector Store")
    result = create_vector_store(request_id, splitted_docs)

    if result:
        st.write("Hurray!! PDF processed successfully")
    else:
        st.write("Error!! Please check logs.")




def main():
    st.header("Divyansh PDF chat using Bedrock")