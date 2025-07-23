from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings.documents_loader import documents as documents_list

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
)

print("Chunking Started......")
chunks = text_splitter.split_documents(documents_list)
print(f"chunk type: {type(chunks[0])}")
print(f"sample chunk metadata: {chunks[0].metadata}\n")
print(f"chunks Len: {len(chunks)}")

for index, chunk in enumerate(chunks):
    if "page" in chunk.metadata:
        chunk.metadata["id"] = (
            f"{chunk.metadata['source']}:{chunk.metadata.get('page')}:{index}"
        )
    else:
        chunk.metadata["id"] = f"{chunk.metadata['source']}:{index}"

print(f"chunk[0].metadata['id']: {chunks[0].metadata['id']}\n")
