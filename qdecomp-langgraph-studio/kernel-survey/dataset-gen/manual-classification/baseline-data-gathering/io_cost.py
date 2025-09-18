


# Prices per 1M tokens for different models on OpenRouter
# (input_cost, output_cost)
MODEL_PRICES = {
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4o-mini": (0.15, 0.6),
    "o4-mini-high": (1.1, 4.4),
    "o4-mini": (1.1, 4.4),
    "o3-mini-high": (1.1, 4.4),
    "o3-mini": (1.1, 4.4),
    "gemini-flash-1.5": (0.075, 0.3),
    "gemini-2.0-flash-lite-001": (0.075, 0.3),
    "gemini-2.0-flash-001": (0.1, 0.4),
    "gemini-2.5-flash": (0.3, 2.5),
    "gemini-2.5-flash-lite": (0.1, 0.4),
    "claude-3.5-haiku": (0.8, 4.0),
    "gpt-5-mini": (0.25, 2.0)
}

def get_io_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    for model in MODEL_PRICES.keys():
        if model in model_name:
            input_cost_per_mil, output_cost_per_mil = MODEL_PRICES[model]
            cost = ((input_tokens / 1e6) * input_cost_per_mil) + \
                   ((output_tokens / 1e6) * output_cost_per_mil)
            return cost

    return -1.0

def get_query_cost(response, verbose: bool = False):
    metadata = response.response_metadata
    token_usage = metadata.get("token_usage", {})
    model_name = metadata.get("model_name", "unknown")

    if verbose:
        print('Token usage metadata:', metadata, flush=True)
        print('Token usage:', token_usage, flush=True)
        print(f'Model name: [{model_name}]', flush=True)

    input_tokens = token_usage.get("prompt_tokens", -1)
    output_tokens = token_usage.get("completion_tokens", -1)

    io_cost = get_io_cost(model_name, input_tokens, output_tokens) 

    if verbose and io_cost < 0:
        # produce a WARNING if the model is unknown
        print(f"\tWARNING: Unknown model name '{model_name}'. Cannot compute I/O cost correctly!", flush=True)

    if verbose:
        print("This cost:", io_cost, flush=True)
        print("Input tokens for this query:", input_tokens, flush=True)
        print("Output tokens for this query:", output_tokens, flush=True)

    return {"input_tokens": [input_tokens],
            "output_tokens": [output_tokens],
            "total_cost": [io_cost]}
