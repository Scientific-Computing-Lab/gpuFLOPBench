from pydantic import BaseModel, Field
from typing import Annotated, Literal
from .dataset import target_names

llm_nodes = [
            "float_div_check_1", 
]

class Configuration(BaseModel):
    temp : float = Field(default=0.2, 
                         description="The temperature to use for the LLM. Higher values make the output more random, lower values make it more deterministic.",
                         json_schema_extra={"langgraph_nodes": llm_nodes}
                         )

    top_p : float = Field(default=0.1, 
                          description="The top_p value to use for the LLM. Higher values make the output more random, lower values make it more deterministic. This is used in conjunction with temperature to control the randomness of the output.",
                          json_schema_extra={"langgraph_nodes": llm_nodes}
                          )

    provider_url : str = Field(default="https://openrouter.com/api/v1",
                              description="The URL of the provider's API endpoint. This is used to connect to the LLM provider.",
                              json_schema_extra={"langgraph_nodes": llm_nodes}
                              )

    provider_api_key: str = Field(default="",
                                 description="The API key for the LLM provider. This is used to authenticate requests to the provider's API.",
                                 json_schema_extra={"langgraph_nodes": llm_nodes}
                                 )

    model: Annotated[
        Literal[
            "openai/gpt-4.1-nano", # in $0.1 out $0.4
            "openai/gpt-4.1-mini", # in $0.4 out $1.6
            "openai/gpt-4o-mini", # in $0.15 out $0.6
            "openai/o4-mini-high", # in $1.1 out $4.4
            "openai/o4-mini", # in $1.1 out $4.4
            "openai/o3-mini-high", # in $1.1 out $4.4
            "openai/o3-mini", # in $1.1 out $4.4
            "google/gemini-flash-1.5", # in $0.075 out $0.3
            "google/gemini-2.0-flash-lite-001", # in $0.075 out $0.3
            "google/gemini-2.0-flash-001", # in $0.1 out $0.4
            "google/gemini-2.5-flash", # in $0.3 out $2.5
            "anthropic/claude-3.5-haiku" # in $0.8 out $4.0
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="openai/gpt-4.1-mini",
        #default="openai/o3-mini",
        description="The name of the language model to use for the agent's main interactions. "
        "Should be in the form: provider/model-name.",
        json_schema_extra={"langgraph_nodes": llm_nodes},
    )


    input_problem : Annotated[
        Literal[
            *target_names
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="resize-cuda",
        description="The name of the input CUDA program to study.",
        json_schema_extra={"langgraph_nodes": ["get_input_problem_0"]},
    )