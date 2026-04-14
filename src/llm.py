from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import config


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )

def get_gemini_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0,
    )

def rewrite_query(query: str) -> str:
    # llm = get_gemini_llm() 
    llm = get_llm() # open ai llm
    prompt = f"""
Rewrite this user query only if it is ambiguous for retrieval.
Otherwise return it unchanged.

Query: {query}
"""
    return llm.invoke(prompt).content.strip()


def generate_answer(query: str, contexts: list[str]) -> str:
    llm = get_llm()
    joined_context = "\n\n---\n\n".join(contexts)

    prompt = f"""
You are a grounded QA assistant.
Answer only from the provided context.
If the context is insufficient, say that clearly.

User question:
{query}

Context:
{joined_context}
"""
    return llm.invoke(prompt).content.strip()


def fallback_answer(query: str) -> str:
    llm = get_llm()
    prompt = f"""
The retrieval pipeline could not find strong supporting context.

Respond to the user's question conservatively.
Say that the answer may be incomplete due to weak retrieval.

Question: {query}
"""
    return llm.invoke(prompt).content.strip()