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

Before writing the final response, internally identify the strongest supporting
note(s) or snippets from the retrieved context, including note_id/source when
available. Use that evidence to decide whether the answer is supported.

Do NOT include an Evidence section in the final response. The application shows
retrieved notes separately in the Evidence Vault.

When answering, follow this format exactly. The answer should be clinically
specific, not generic: include the key diagnoses, mechanisms, imaging/lab
findings, symptoms, procedures, treatments, and follow-up details that are
directly supported by the retrieved notes. If several retrieved notes support
the same conclusion, synthesize them into one detailed paragraph. Do not quote
or list evidence snippets in the final response.

Answer:
<give a detailed answer based only on the retrieved context; usually 4-8 sentences>

Confidence:
<High / Medium / Low>
<brief reason based on the number, specificity, and consistency of supporting retrieved notes, without quoting evidence>

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

Confidence:
Low
Reason: retrieval returned weak or insufficient supporting evidence.

User question:
{query}
"""
    return llm.invoke(prompt).content.strip()


def summarize_clinical_context(user_query: str, context: str) -> str:
    llm = get_llm()

    prompt = f"""
You are assisting a physician.

Use the notes below to answer the request.

User request:
{user_query}

Clinical notes:
{context}

Provide a concise structured summary covering:
- Key conditions
- Hospital course
- Important treatments/procedures
- Relevant psychiatric/social history if clinically relevant
- Discharge status / follow-up if available

If information is missing, say so.
"""

    return llm.invoke(prompt).content

def summarize_patient_notes(context: str) -> str:
    llm = get_llm()

    prompt = f"""
You are assisting a physician.

These are multiple notes for one patient.

Create a concise patient summary covering:

- Major diagnoses / chronic conditions
- Relevant psychiatric / social history
- Important admissions or presentations
- Treatments / procedures
- Clinical trajectory over time
- Current risks or follow-up needs if mentioned

Be concise and clinically useful.

Notes:
{context}
"""

    return llm.invoke(prompt).content


def summarize_single_note(context: str) -> str:
    llm = get_llm()

    prompt = f"""
You are assisting a physician.

Summarize this clinical note clearly.

Include:

- Why patient presented
- Important findings
- Treatments / procedures
- Key diagnoses
- Disposition / follow-up

Note:
{context}
"""

    return llm.invoke(prompt).content


def summarize_visit_notes(context: str) -> str:
    llm = get_llm()

    prompt = f"""
You are assisting a physician.

These notes belong to one hospitalization / visit.

Summarize:

- Reason for admission
- Hospital course
- Procedures / treatments
- Key findings
- Discharge condition
- Follow-up needs

Notes:
{context}
"""

    return llm.invoke(prompt).content
