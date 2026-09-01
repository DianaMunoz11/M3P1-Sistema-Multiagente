"""
Construye (o carga desde disco) los índices vectoriales FAISS de cada
dominio de conocimiento: RR. HH. y Tecnología.

En un sistema real, cada dominio suele tener su propio pipeline de
ingesta (fuente de datos, chunking, actualización periódica). Aquí se
simplifica cargando dos archivos de texto de ejemplo.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDINGS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

os.makedirs(INDEX_DIR, exist_ok=True)

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "],
)


def _build_or_load(domain_name: str, source_file: str) -> FAISS:
    """Carga el índice FAISS de un dominio si ya existe; si no, lo construye."""
    index_path = os.path.join(INDEX_DIR, domain_name)

    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, EMBEDDINGS, allow_dangerous_deserialization=True
        )

    loader = TextLoader(os.path.join(DATA_DIR, source_file), encoding="utf-8")
    raw_docs = loader.load()
    chunks = SPLITTER.split_documents(raw_docs)

    for chunk in chunks:
        chunk.metadata["domain"] = domain_name

    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
    vectorstore.save_local(index_path)
    return vectorstore


def get_hr_retriever(k: int = 4):
    """Retriever del dominio RR. HH."""
    vectorstore = _build_or_load("rr_hh", "hr_docs.txt")
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_tech_retriever(k: int = 4):
    """Retriever del dominio Tecnología."""
    vectorstore = _build_or_load("tecnologia", "tech_docs.txt")
    return vectorstore.as_retriever(search_kwargs={"k": k})
