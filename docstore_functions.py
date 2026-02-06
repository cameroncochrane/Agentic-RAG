import langchain_core
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)

from sentence_transformers import SentenceTransformer

import os
from typing import List, Union, Optional
from __future__ import annotations
import hashlib

def scan_directory(directory_path: str):
    """
    Scan a given directory and return the paths of all files inside.

    Args:
        directory_path (str): The path of the directory to scan.

    Returns:
        List[str]: A list of file paths.
    """
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# wrapper around sentence_transformers to match LangChain embeddings interface
class SentenceTransformerEmbeddingsWrapper:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", lazy: bool = True):
        # lazy=True avoids downloading the model at import/definition time.
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.lazy = lazy
        if not self.lazy:
            self._ensure_model()

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of documents.
        This matches LangChain's `embed_documents` contract.
        """
        self._ensure_model()
        emb = self._model.encode(texts, show_progress_bar=False)
        return emb.tolist()

    def embed_query(self, text: str) -> List[float]:
        self._ensure_model()
        emb = self._model.encode([text], show_progress_bar=False)
        return emb[0].tolist()

    def __call__(self, texts: Union[str, List[str]]):
        # Support both single-query strings and lists of documents.
        if isinstance(texts, str):
            return self.embed_query(texts)
        if isinstance(texts, (list, tuple)):
            return self.embed_documents(list(texts))
        raise TypeError(f"Unsupported input type for embeddings: {type(texts)}")
    
def create_faiss_store_from_documents(documents: List[Document], index_dir: str = "vectorstore", embedding_model: str = "BAAI/bge-small-en-v1.5"):
    """
    Build a FAISS vectorstore from a list of langchain Document objects and save it to disk.
    Returns the in-memory FAISS store.
    """
    os.makedirs(index_dir, exist_ok=True)
    embeddings = SentenceTransformerEmbeddingsWrapper(embedding_model)
    # Pass the embeddings wrapper object so FAISS can access its
    # `embed_documents` / `embed_query` methods as expected.
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(index_dir)
    return store

def load_faiss_store(index_dir: str = "vectorstore", embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Load a previously saved FAISS vectorstore from disk.
    """
    embeddings = SentenceTransformerEmbeddingsWrapper(embedding_model)
    # FAISS.load_local expects an embeddings object providing `embed_documents`
    return FAISS.load_local(index_dir, embeddings)

def add_documents_and_save(store: FAISS, new_documents: List[Document], index_dir: str = "vectorstore"):
    """
    Add documents to an existing FAISS store and persist to disk.
    """
    store.add_documents(new_documents)
    store.save_local(index_dir)
    return store


def path_upload_document_to_vectorstore(
    document_paths: Union[str, List[str]],
    store: FAISS,
    index_dir: str = "vectorstore",
    dedup_mode: str = "content",  # "content" (recommended) or "source"
):
    """
    Upload one or more documents to an existing FAISS vectorstore and persist to disk
    Ensures each Document has a `content_hash` in metadata so subsequent
    content-based deduplication works even for the initial ingestion.
    WITHOUT erasing existing contents, with deduplication.

    # Ensure documents have a content_hash for idempotent ingestion
    def _content_hash(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    for d in documents:
        # only operate on Document instances
        if not isinstance(d, Document):
            continue
        h = d.metadata.get("content_hash")
        if not h:
            h = _content_hash(d.page_content)
            d.metadata["content_hash"] = h
        # ensure a source field exists for traceability
        d.metadata.setdefault("source", d.metadata.get("source", ""))
    Deduplication behavior:
      - dedup_mode="content": hashes each loaded Document.page_content and skips exact duplicates.
      - dedup_mode="source": skips if an existing Document has the same metadata["source"].

    Notes:
      - For PDFs, PyPDFLoader returns one Document per page; dedup will happen at page level.
      - This function assumes your loaders populate metadata["source"] (LangChain usually does).
    """
    if isinstance(document_paths, str):
        document_paths = [document_paths]
    
    def _content_hash(text: str) -> str:
        # stable content-based id
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


    def _get_existing_hashes(store: FAISS) -> set[str]:
        """
        Extract content hashes from the existing LangChain FAISS docstore.
        We store hashes in Document.metadata["content_hash"] for idempotent ingestion.
        """
        existing = set()

        # LangChain FAISS keeps docs in an InMemoryDocstore at store.docstore._dict
        doc_dict = getattr(getattr(store, "docstore", None), "_dict", None)
        if isinstance(doc_dict, dict):
            for d in doc_dict.values():
                if isinstance(d, Document):
                    h = d.metadata.get("content_hash")
                    if h:
                        existing.add(h)
        return existing

    # Build the dedup index from the current store
    existing_hashes = _get_existing_hashes(store) if dedup_mode == "content" else set()

    existing_sources = set()
    if dedup_mode == "source":
        doc_dict = getattr(getattr(store, "docstore", None), "_dict", None)
        if isinstance(doc_dict, dict):
            for d in doc_dict.values():
                if isinstance(d, Document):
                    src = d.metadata.get("source")
                    if src:
                        existing_sources.add(src)

    total_added = 0
    total_skipped = 0

    for document_path in document_paths:
        # Determine the loader type based on file extension
        if document_path.lower().endswith(".pdf"):
            loader_cls = PyPDFLoader
        elif document_path.lower().endswith(".txt"):
            loader_cls = TextLoader
        elif document_path.lower().endswith(".md"):
            loader_cls = UnstructuredMarkdownLoader
        else:
            print(f"Unsupported file type for {document_path}. Supported types are: .pdf, .txt, .md")
            continue

        # Load the document(s)
        try:
            loader = loader_cls(document_path)
            loaded_docs = loader.load()
            print(f"Loaded {len(loaded_docs)} document(s) from {document_path}.")
        except Exception as e:
            print(f"Error loading document {document_path}: {e}")
            continue

        # Tag documents with dedup metadata and filter duplicates
        new_docs: List[Document] = []
        for d in loaded_docs:
            # Ensure source is set for traceability (helps source-based dedup & citations)
            d.metadata.setdefault("source", document_path)

            if dedup_mode == "source":
                src = d.metadata.get("source")
                if src in existing_sources:
                    total_skipped += 1
                    continue
                existing_sources.add(src)
                new_docs.append(d)
                continue

            # content-based dedup (recommended)
            h = d.metadata.get("content_hash")
            if not h:
                h = _content_hash(d.page_content)
                d.metadata["content_hash"] = h

            if h in existing_hashes:
                total_skipped += 1
                continue

            existing_hashes.add(h)
            new_docs.append(d)

        if not new_docs:
            print(f"No new chunks/pages to add from {document_path} (all duplicates).")
            continue

        # Add to store and persist
        store.add_documents(new_docs)
        store.save_local(index_dir)
        total_added += len(new_docs)

        print(f"Added {len(new_docs)} new document(s) from {document_path} and saved to '{index_dir}'.")

    print(f"Done. Added: {total_added}, skipped as duplicates: {total_skipped}.")
    return store

def load_docstore_from_dir(index_dir: str = "vectorstore", embedding_model: str = "BAAI/bge-small-en-v1.5"):
    """
    Load a FAISS-backed docstore from disk and return (store, documents_list).
    """
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Index directory '{index_dir}' not found.")

    embeddings = SentenceTransformerEmbeddingsWrapper(embedding_model)
    try:
        store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS store from '{index_dir}': {e}")

    doc_dict = getattr(getattr(store, "docstore", None), "_dict", None) or {}
    docs = [d for d in doc_dict.values() if isinstance(d, Document)]

    print(f"Loaded FAISS store from '{index_dir}' with {len(docs)} document(s).")
    return store, docs

from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS




def retrieve_local(store: FAISS, query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Returns normalized chunks for use by agents/LLM.
    """
    retriever = store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    out = []
    for i, d in enumerate(docs, start=1):
        out.append({
            "id": f"L-{i:04d}",
            "text": d.page_content,
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", None),
            "content_hash": d.metadata.get("content_hash", None),
        })
    return out
