from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import config


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.OPENAI_EMBED_MODEL,
        api_key=config.OPENAI_API_KEY,
    )



# def get_embedding_model():
#     return GoogleGenerativeAIEmbeddings(
#         model=config.GEMINI_EMBED_MODEL,
#         google_api_key=config.GEMINI_API_KEY,
#     )