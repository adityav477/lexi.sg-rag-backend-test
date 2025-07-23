from pathlib import Path
import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader,
)

folder = Path(__file__).resolve().parent.parent / "documents"


def get_file_loader(file_path: str):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif file_path.endswith(".docs") or file_path.endswith(".doc"):
        return UnstructuredWordDocumentLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path)


def file_loaders():
    documents = []
    for root, _, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            print(f"Loading: {full_path}")
            try:
                loader = get_file_loader(full_path)
                docs = loader.load()
                # print(f" -> Loaded {len(docs)} docs from {len(file)}")
                documents.extend(docs)
            except Exception as e:
                print(f"failed to load documents {full_path} error -> {e}")

    return documents


# list with elemnt of type : List with each element of this list being type: object document which itslef has attributes of metadata and page_content
documents = file_loaders()
print(f"Loaded {len(documents)} documents")
print(f"sample doc: {documents[0].metadata}\n")
print(f"sample doc2: {documents[1].metadata}\n")
print(f"last doc2: {documents[201].metadata}")

# for document.append
# # print(f"docs: {documents[0][:100]}")
# print(f"docs: {type(documents[0][0].page_content)} \n")
# # print(f"documents[0]: {final_text[0]} \n")
# print(
#     f"documents[0][0].metadata.get['source']: {documents[0][0].metadata.get('source')} \n"
# )
# print(f"documents[0][0].metadata type: {type(documents[0][0].metadata)} \n")
# print(f"text: {documents[2][0].page_content[:100]} \n")
#
# # list with elemnt of type : object document which itslef has attributes of metadata and page_content
# final_doc = [doc[0] for doc in documents]
# print(f"fina_doc.len: {len(final_doc)}\n")
# # print(f"fina_doc[0]:{final_doc[0]}\n")
# print(f"sameple page_conent: {final_doc[0].page_content[:100]}\n")
# print(f"fina_doc[0] metadata:{final_doc[0].metadata}\n")
