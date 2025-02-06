import streamlit as st
import fitz
import docx
from io import BytesIO
import pytesseract
from pdf2image import convert_from_path
import os
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import asyncio
from difflib import SequenceMatcher
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Set up cache for the language model
set_llm_cache(InMemoryCache())

SUPPORTED_FILES = ["txt", "pdf", "docx"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_BASE_URL = "http://localhost:11434"  # Or your Ollama URL

# Initialize embeddings and text splitter
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP#, separators=["\n\n", "\n", ". ", " ", "-"]
)
llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model="llama3.2")

# Function to create vector database from text chunks
def create_vector_db(text):
    chunks = text_splitter.split_text(text)
    db = FAISS.from_texts(chunks, embeddings)
    return db, chunks

# Function to extract text from image-based PDFs using OCR
def extract_text_from_image(pdf_bytes):
    images = convert_from_path(pdf_bytes)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Function to read file content based on file type
def read_file_content(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'txt':
            # Read content of .txt file
            return uploaded_file.getvalue().decode('utf-8')
        elif file_type == 'pdf':
            # Read content of .pdf file
            text = ""
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            for page_num, page in enumerate(doc):
                extracted_text = page.get_text("text")
                if extracted_text:
                    text += extracted_text.replace('\n', ' ').strip() + "\n\n"
                else:
                    print(f"No text found on page {page_num + 1}.")
            if not text:
                print(f"Warning: No text extracted from the PDF.")
                # Fallback to OCR if no text is extracted
                text = extract_text_from_image(uploaded_file.getvalue())
            return text.strip() if text else None
        elif file_type in ['doc', 'docx']:
            # Read content of .docx file
            doc = docx.Document(BytesIO(uploaded_file.getvalue()))
            return "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
        else:
            print(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Function to highlight word differences in text
def highlight_word_differences(orig_text, new_text):
    matcher = SequenceMatcher(None, orig_text.split(), new_text.split())
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            result.append(f'<span style="color: yellow; text-decoration: line-through;">{" ".join(orig_text.split()[i1:i2])}</span> ')
            result.append(f'<span style="color: lightgreen; font-weight: bold;">{" ".join(new_text.split()[j1:j2])}</span> ')
        elif tag == 'insert':
            result.append(f'<span style="color: lightblue; font-weight: bold;">{" ".join(new_text.split()[j1:j2])}</span> ')
        elif tag == 'delete':
            result.append(f'<span style="color: red; text-decoration: line-through;">{" ".join(orig_text.split()[i1:i2])}</span> ')
        else:
            result.append(f'{" ".join(orig_text.split()[i1:i2])} ')
    return "".join(result)

# Function to extract and display differences
def extract_differences(text1, text2):
    return highlight_word_differences(text1, text2)

# Async function for chatbot
async def get_answer_async(query, combined_db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=combined_db.as_retriever())
    answer = await qa_chain.arun(query)  # Async call to LLM
    return answer

# Main function to run Streamlit app
def main():
    st.title("Document Comparison and Chatbot - RAG")

    # Initialize session state variables
    if 'doc1_content' not in st.session_state:
        st.session_state.doc1_content = None
    if 'doc2_content' not in st.session_state:
        st.session_state.doc2_content = None
    if 'doc1_db' not in st.session_state:
        st.session_state.doc1_db = None
    if 'doc2_db' not in st.session_state:
        st.session_state.doc2_db = None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Document 1")
        file1 = st.file_uploader("Upload Document 1", type=SUPPORTED_FILES, key="file1")
        if file1:
            with st.spinner("Processing..."):
                st.session_state.doc1_content = read_file_content(file1)
                if st.session_state.doc1_content:
                    st.session_state.doc1_db, st.session_state.doc1_chunks = create_vector_db(st.session_state.doc1_content)  # Store chunks
                    st.success("Document 1 processed!")

    with col2:
        st.subheader("Document 2")
        file2 = st.file_uploader("Upload Document 2", type=SUPPORTED_FILES, key="file2")
        if file2:
            with st.spinner("Processing..."):
                st.session_state.doc2_content = read_file_content(file2)
                if st.session_state.doc2_content:
                    st.session_state.doc2_db, st.session_state.doc2_chunks = create_vector_db(st.session_state.doc2_content)  # Store chunks
                    st.success("Document 2 processed!")

    if st.session_state.doc1_content and st.session_state.doc2_content:
        st.subheader("Document Comparison")

        st.subheader("Chatbot")
        chatbot_query = st.text_area("Ask a question about the documents:", height=100)
        if st.button("Ask Chatbot"):
            if st.session_state.doc1_db and st.session_state.doc2_db:
                with st.spinner("Generating response..."):
                    all_texts = st.session_state.doc1_chunks + st.session_state.doc2_chunks
                    combined_db = FAISS.from_texts(all_texts, embeddings)

                    # Async call
                    answer = asyncio.run(get_answer_async(chatbot_query, combined_db))
                    st.write(answer)
            else:
                st.warning("Please upload and process both documents first.")

    if st.button("Compare"):
        with st.spinner("Comparing..."):
            doc1_chunks = text_splitter.split_text(st.session_state.doc1_content)
            doc2_chunks = text_splitter.split_text(st.session_state.doc2_content)

            doc1_embeddings = np.array([embeddings.embed_query(chunk) for chunk in doc1_chunks])
            doc2_embeddings = np.array([embeddings.embed_query(chunk) for chunk in doc2_chunks])

            index = faiss.IndexFlatL2(doc2_embeddings[0].shape[0])
            index.add(doc2_embeddings.astype('float32'))

            all_diffs_html = ""  # Accumulate ONLY the diff HTML

            for i, doc1_embedding in enumerate(doc1_embeddings):
                D, I = index.search(np.array([doc1_embedding]).astype('float32'), k=3)
                most_similar_doc2_index = I[0][0]

                chunk_diff_html = extract_differences(doc1_chunks[i], doc2_chunks[most_similar_doc2_index])

                all_diffs_html += chunk_diff_html

            st.write(f"{all_diffs_html}", unsafe_allow_html=True)  # Single outer div

if __name__ == "__main__":
    main()
