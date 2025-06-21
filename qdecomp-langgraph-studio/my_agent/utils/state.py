from typing_extensions import TypedDict, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

#class AgentState(TypedDict):
#    messages: Annotated[Sequence[BaseMessage], add_messages]

# Class used to represent a warp divergence point in the kernel source code
# Part of step 7a, where we extract the warp divergence points from the annotated kernel source code
class WarpDivergencePoint(BaseModel):
    classification: str = Field(..., description="Derived classification of the warp divergence point, which can be one of the following: 'for', 'if', 'else-if', 'while', 'do-while', 'ternary'. This classification is used to classify the type of warp divergence point.")
    source_code: str = Field(..., description="Extracted source code of the warp divergence point, including the conditional logic and necessary variables used in the warp divergence point entry logic. The source code should include the `// WARP DIVERGENCE POINT -- VARIABLES REASONING` comment.")

# Updated KernelAnalysisState using TypedDict with default values for optional fields
class KernelAnalysisState(TypedDict, total=False):
    source_code: str
    kernel_name: str
    exec_args: str
    grid_size: str
    block_size: str
    total_num_threads: str

    # these will be filled in by the nodes
    src_concretized_input_args: str
    src_single_kernel_execution: str
    snippet_first_kernel_invocation: str
    snippet_kernel_src: str
    snippet_kernel_src_concretized_values: str
    kernel_annotated_warp_divergence: str
    kernel_annotated_WDPs: str
    wdps_list: List[WarpDivergencePoint]  # List of tuples with warp divergence point source code and classification
    wdps_num_executions : List[int]

    kernel_annotated_num_ops: str
    summed_kernel_ops: str