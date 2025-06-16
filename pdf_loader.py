import streamlit as st
import tempfile
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# === LLM and embeddings ===
llm = OllamaLLM(model="mistral")
embed = OllamaEmbeddings(model="mistral")

# === Text splitter ===
pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    separators=["\n\n",",","."," ","\n"]
)

# === Prompt ===
template = """
You are an expert resume parser.

Extract the following information from the resume text:
- Education (list degrees with institution and year)
- Skills
- Hobbies

Resume Text:
{context}

Format the output clearly using bullet points and labels like this:


Mail id : 
-Gmail

Education:
- Degree at Institution (Year)
- Degree at Institution (Year)

Skills:
- Skill 1
- Skill 2
- Skill 3


"""


prompt = PromptTemplate.from_template(template)


# === Stuff chain ===
document_chain = create_stuff_documents_chain(llm, prompt)

# === PDF processor function ===
def pdfprocessor(text_chunks):
    vectorstore = Chroma.from_documents(text_chunks, embed, persist_directory="./chroma_store")
    retriever = vectorstore.as_retriever(search_type="mmr")
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": ""})  # We use retriever to find context, no need to pass text
    return response["answer"]

# === Streamlit UI ===
st.title("📄 Resume Extractor with LLaMA2")

uploaded_file = st.file_uploader("Upload a Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    loader = PyPDFLoader(tmp_path)
    docs = loader.load() 
    text_chunks = pdf_splitter.split_documents(docs)

    with st.spinner("🔍 Analyzing resume..."):
        result = pdfprocessor(text_chunks)

    try:
        st.text(result)
    except Exception as e:
        st.error("⚠️ Failed to parse resume content as JSON.")
