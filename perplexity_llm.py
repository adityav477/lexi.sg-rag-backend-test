import getpass
import os
from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
from pydantic import SecretStr
from embeddings.vector_store import vector_store


def load_perplexity_api_key():
    load_dotenv()
    api_key = os.getenv("PPLX_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your PPLX_API_KEY: ")
        os.environ["PPLX_API_KEY"] = api_key

    return SecretStr(api_key)


def get_answer(query: str):
    context_chunks = vector_store.similarity_search(
        query,
        k=3,
    )

    chat = ChatPerplexity(
        api_key=load_perplexity_api_key(),
        temperature=0,
        model="sonar",
        timeout=0,
        streaming=True,
    )

    local_chat = OllamaLLM(model="steamdj/llama3.1-cpu-only", temperature=0)

    template = """You are a legal assistant trained in Indian law. Your job is to analyze retrieved legal text passages and generate a precise, well-grounded answer to a user's query, including citations to the relevant legal source documents.

    ## Task
    Answer the following legal query based on the retrieved text snippets provided below. Use only the retrieved passages for your answer. Do not invent facts or laws that are not mentioned in the retrieved context.

    ## Output Format
    Respond in the following JSON format:
    {{"answer": "<your generated legal answer>",
    "citations": [
        {{"text": "<exact quoted snippet used in the answer>",
        "source": "<document name or source>"
        }},
      ]
    }}
    
## Legal Query:
    {query}

    ## Retrieved Context (passages from legal documents):
    {context_chunks}
    
## Rules:
    - Only answer based on the context provided.
    - If the query cannot be answered using the given context, say: "Based on the provided documents, there is insufficient information to answer this query."
    - Cite directly quoted snippets from the context used in the answer, and include their source names.

    Respond only in valid JSON.
    """

    prompt = ChatPromptTemplate.from_messages([("system", template), ("human", query)])

    llm = prompt | chat
    local_llm = prompt | local_chat

    response = llm.invoke({"query": query, "context_chunks": context_chunks})

    return response.content
