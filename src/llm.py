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
You are a clinical data analysis assistant.

Your task is to answer the user's question using ONLY the provided clinical note context.
Do NOT use outside medical knowledge.
Do NOT guess.
Do NOT fabricate evidence.
If the answer cannot be supported directly by the retrieved context, do not provide a speculative answer.

If the provided context does not contain enough evidence to answer the question, say:
"Unable to confirm from the available retrieved notes."

When answering, follow this format exactly:

Answer:
<give a concise answer based only on the retrieved context>

Evidence:
<list the most relevant supporting note(s) or short quoted snippets from the retrieved context; include note_id and source whenever available>

Confidence:
<High / Medium / Low>
<brief reason, such as number of supporting notes, consistency of evidence, or limited evidence>

User question:
{query}

Retrieved context:
{joined_context}
"""
    return llm.invoke(prompt).content.strip()


def fallback_answer(query: str) -> str:
    llm = get_llm()
    prompt = f"""
You are a clinical data analysis assistant.

The retrieval pipeline could not find strong supporting evidence for the user's question.

Respond conservatively.
Do NOT fabricate facts.
Do NOT rely on outside medical knowledge.

Use this format exactly:

Answer:
Unable to confirm from the available retrieved notes.

Evidence:
No strong supporting context was retrieved.

Confidence:
Low
Reason: retrieval returned weak or insufficient supporting evidence.

User question:
{query}
"""
    return llm.invoke(prompt).content.strip()
