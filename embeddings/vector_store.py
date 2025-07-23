from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./vector_store_langchainchroma"
COLLECTION_NAME = "mychromadb2"

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
    persist_directory=PERSIST_DIR,
)


# cheks if the store local store is empty or not
def vector_store_is_empty(vs: Chroma) -> bool:
    try:
        results = vs.similarity_search("test", k=1)
        return len(results) == 0
    except Exception:
        return True


if vector_store_is_empty(vector_store):
    print("Vector store is empty, loading chunks and adding...")
    from embeddings.embedder import chunks

    if not chunks:
        raise ValueError("Chunks are empty — check your embedding logic.")

    vector_store.add_documents(
        ids=[chunk.metadata["id"] for chunk in chunks],
        documents=chunks,
    )
    print("Documents added and stored.")
else:
    print("Vector store already contains data — skipping add_documents.")
