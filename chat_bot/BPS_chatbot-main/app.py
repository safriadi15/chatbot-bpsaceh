import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(".env")


# Display logo and header
logo_path = "data/bps.png"

# Check if the logo file exists
if os.path.exists(logo_path):
    st.image(logo_path, width=150)  # Adjust width as needed
else:
    st.warning("Logo file not found. Make sure 'bps_logo.png' is in the same directory as your script.")

st.title("BADAN PUSAT STATISTIK PROVINSI ACEH")

# Replace file uploader with direct file path
pdf_path = "data/ProfilBPS.pdf"  # Ganti dengan path ke file PDF Anda

if os.path.exists(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split menjadi chunk
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
else:
    st.error("File PDF tidak ditemukan. Pastikan path file sudah benar 😺.")

pertanyaan = st.text_input("Perlu info BPS Provinsi Aceh tanya aja!!!👋")
if pertanyaan:
    # Ensure that knowledge_base is initialized
    if 'knowledge_base' in locals():
        docs = knowledge_base.similarity_search(pertanyaan)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type='stuff')
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=pertanyaan)
        st.subheader("Jawaban:")
        st.write(response)
    else:
        st.error("Knowledge base belum terinisialisasi. Pastikan file PDF sudah benar dan diproses.")

