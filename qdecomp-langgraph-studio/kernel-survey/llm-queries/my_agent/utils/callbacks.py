from langchain_core.callbacks import BaseCallbackHandler
from typing import Dict, Any, List
from langchain_core.outputs import LLMResult
from .state import KernelAnalysisState

# Prices per 1M tokens for different models on OpenRouter
# (input_cost, output_cost)
MODEL_PRICES = {
    "openai/gpt-4.1-nano": (0.1, 0.4),
    "openai/gpt-4.1-mini": (0.4, 1.6),
    "openai/gpt-4o-mini": (0.15, 0.6),
    "openai/o4-mini-high": (1.1, 4.4),
    "openai/o4-mini": (1.1, 4.4),
    "openai/o3-mini-high": (1.1, 4.4),
    "openai/o3-mini": (1.1, 4.4),
    "google/gemini-flash-1.5": (0.075, 0.3),
    "google/gemini-2.0-flash-lite-001": (0.075, 0.3),
    "google/gemini-2.0-flash-001": (0.1, 0.4),
    "google/gemini-2.5-flash": (0.3, 2.5),
    "google/gemini-2.5-flash-lite": (0.1, 0.4),
    "anthropic/claude-3.5-haiku": (0.8, 4.0),
    "gpt-5-mini": (0.25, 2.0)
}

def get_io_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    if model_name in MODEL_PRICES:
        input_cost_per_mil, output_cost_per_mil = MODEL_PRICES[model_name]
        cost = ((input_tokens / 1e6) * input_cost_per_mil) + \
               ((output_tokens / 1e6) * output_cost_per_mil)
        return cost
    return -1.0

#def update_state_costs(state:KernelAnalysisState, response):
def get_query_cost(response, verbose: bool = False):
    #if "input_tokens" not in state:
    #    state["input_tokens"] = 0
    #if "output_tokens" not in state:
    #    state["output_tokens"] = 0
    #if "total_cost" not in state:
    #    state["total_cost"] = 0.0

    metadata = response.response_metadata
    token_usage = metadata.get("token_usage", {})
    model_name = metadata.get("model_name", "unknown")

    #print('Token usage metadata:', metadata)
    #print('Token usage:', token_usage)
    #print('Model name:', model_name)

    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)

    #state["input_tokens"] += input_tokens
    #state["output_tokens"] += output_tokens

    io_cost = get_io_cost(model_name, input_tokens, output_tokens) 
    #state["total_cost"] += io_cost

    if verbose:
        print("This cost:", io_cost)
        print("Input tokens for this query:", input_tokens)
        print("Output tokens for this query:", output_tokens)

    return {"input_tokens": [input_tokens],
            "output_tokens": [output_tokens],
            "total_cost": [io_cost]}
