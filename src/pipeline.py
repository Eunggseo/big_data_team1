from src.state import PipelineState
from src.agent import build_graph
from src.logger import log_run

graph = build_graph()


def run_pipeline(query: str) -> PipelineState:
    initial_state = PipelineState(query=query)
    result = graph.invoke(initial_state)
    final_state = PipelineState(**result)

    log_run(final_state)   

    return final_state