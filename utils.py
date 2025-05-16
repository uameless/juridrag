import tempfile
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document

from pdf2image import convert_from_bytes
import pytesseract

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        docs = PyMuPDFLoader(tmp_path).load()
        if any(doc.page_content.strip() for doc in docs):
            return docs
    except Exception:
        pass 

    images = convert_from_bytes(open(tmp_path, 'rb').read())
    texts = [pytesseract.image_to_string(img, lang='ara') for img in images]
    return [Document(page_content=text.strip()) for text in texts if text.strip()]


def split_docs(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", "،", " "]
    )
    return splitter.split_documents(documents)

def load_arabic_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

def create_vector_store(texts, embeddings):
    return FAISS.from_documents(texts, embeddings)