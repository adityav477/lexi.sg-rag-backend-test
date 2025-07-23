from fastapi import FastAPI

from perplexity_llm import get_answer as get_answer_perplexity_llm

app = FastAPI()


@app.get("/")
def default():
    return {"message": "Hello World"}


@app.post("/query")
async def queryfunc(query: str):
    response = get_answer_perplexity_llm(query)

    print(response)
    return response
