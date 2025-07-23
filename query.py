from embeddings.vector_store import vector_store
from perplexity_llm import get_answer as get_answer_perplexity_llm

import pprint as pp


def test_embeddings():
    results = vector_store.similarity_search(
        "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?",
        k=2,
    )

    pp.pprint(f"results of test_embeddings: {results}")


def test_response_perplexity():
    response = get_answer_perplexity_llm(
        "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
    )

    pp.pprint(f"result for response from llm: {response}")


if __name__ == "__main__":
    test_response_perplexity()
