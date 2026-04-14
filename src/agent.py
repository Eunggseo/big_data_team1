from langgraph.graph import StateGraph, END
from src.state import PipelineState
from src.cache import get_cached_response, store_response
from src.retriever import retrieve
from src.reranker import rerank
from src.llm import rewrite_query, generate_answer, fallback_answer
from src.evaluator import evaluate_retrieval, evaluate_rerank
from src.config import config


def analyze_and_rewrite(state: PipelineState) -> PipelineState:
    rewritten = rewrite_query(state.query)
    state.rewritten_query = rewritten
    state.debug["rewritten_query"] = rewritten
    return state


def check_cache(state):
    key = (state.rewritten_query or state.query).strip().lower()

    print("LOOKUP KEY:", repr(key))  # debug

    cached = get_cached_response(key)
    if cached:
        state.cache_hit = True
        state.answer = cached

    return state


def retrieve_docs(state: PipelineState) -> PipelineState:
    query = state.rewritten_query or state.query
    k = config.TOP_K + state.retry_count * 2
    docs = retrieve(query, k=k)
    state.retrieved_docs = docs
    state.retrieval_good = evaluate_retrieval(docs)
    state.debug["retrieved_k"] = k
    return state


def rerank_docs(state: PipelineState) -> PipelineState:
    query = state.rewritten_query or state.query
    reranked = rerank(query, state.retrieved_docs, config.TOP_N)
    state.reranked_docs = reranked
    state.rerank_good = evaluate_rerank(reranked)
    return state


def maybe_retry(state: PipelineState) -> PipelineState:
    if (not state.retrieval_good or not state.rerank_good) and state.retry_count < config.MAX_RETRIES:
        state.retry_count += 1
        state.rewritten_query = rewrite_query(
            f"Rewrite this query to improve retrieval specificity: {state.query}"
        )
    return state


def generate(state: PipelineState) -> PipelineState:
    contexts = [d["content"] for d in state.reranked_docs]
    state.answer = generate_answer(state.query, contexts)
    return state


def fallback(state: PipelineState) -> PipelineState:
    state.used_fallback = True
    state.answer = fallback_answer(state.query)
    return state


def save_cache(state):
    key = (state.rewritten_query or state.query).strip().lower()

    print("STORE KEY:", repr(key))  # debug

    if state.answer and not state.cache_hit:
        store_response(key, state.answer)

    return state


def should_continue_after_cache(state: PipelineState) -> str:
    return "end" if state.cache_hit else "retrieve"


def should_retry_or_generate(state: PipelineState) -> str:
    if state.rerank_good:
        return "generate"
    if state.retry_count < config.MAX_RETRIES:
        return "retry"
    return "fallback"


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("rewrite", analyze_and_rewrite)
    graph.add_node("cache", check_cache)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("rerank", rerank_docs)
    graph.add_node("retry", maybe_retry)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
    graph.add_node("save_cache", save_cache)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "cache")

    graph.add_conditional_edges(
        "cache",
        should_continue_after_cache,
        {
            "retrieve": "retrieve",
            "end": END,
        }
    )

    graph.add_edge("retrieve", "rerank")

    graph.add_conditional_edges(
        "rerank",
        should_retry_or_generate,
        {
            "generate": "generate",
            "retry": "retry",
            "fallback": "fallback",
        }
    )

    graph.add_edge("retry", "retrieve")
    graph.add_edge("generate", "save_cache")
    graph.add_edge("fallback", "save_cache")
    graph.add_edge("save_cache", END)

    return graph.compile()