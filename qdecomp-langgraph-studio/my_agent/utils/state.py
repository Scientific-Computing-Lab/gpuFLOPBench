from typing_extensions import TypedDict, List, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from langgraph.graph.message import add_messages

#class AgentState(TypedDict):
#    messages: Annotated[Sequence[BaseMessage], add_messages]

# Class used to represent a warp divergence point in the kernel source code
# Part of step 7a, where we extract the warp divergence points from the annotated kernel source code
class WarpDivergencePoint(BaseModel):
    classification: str = Field(..., description="Derived classification of the warp divergence point, which can be one of the following: 'for', 'if', 'else-if', 'while', 'do-while', 'ternary'. This classification is used to classify the type of warp divergence point.")
    source_code: str = Field(..., description="Extracted source code of the warp divergence point, including the conditional logic and necessary variables used in the warp divergence point entry logic. The source code should include the `// WARP DIVERGENCE POINT -- VARIABLES REASONING` comment.")


# Because this step 7 needs to be more fleshed-out, we create a schema for it to use in
# extracting the WARP DIVERGENCE POINTS and their dependent variables.
class DivergencePointsList(BaseModel):
    # technically we only allow if, else-if, for, and while
    # step 1 forces all the ternary to be converted to if statements
    # and all the do-while loops to while loops,
    # but some may escape through weak models that don't properly transform the code
    # TODO: we need to account for switch case statements....

    """A list of the warp divergence point objects from the kernel source code, with their conditional definition, logic, dependent variables, variables reasoning, and classification. Each warp divergence point is represented as a WarpDivergencePoint object, with a `source_code` string of the warp divergence point, and a `classification` of the warp divergence point from the list: {for, if, else-if, while, do-while, ternary}."""

    warp_divergence_points: List[WarpDivergencePoint] = Field(
        ...,
        description="A list of WarpDivergencePoint objects containing the information about warp divergence points in the kernel source code, where each object contains the source code (source_code) of the warp divergence point and its classification (classification). The classification can be one of the following: 'for', 'if', 'else-if', 'while', 'do-while', 'ternary', and is used to classify the type of warp divergence point. The start of the source code of the warp divergence point should is indicated by the `// WARP DIVERGENCE POINT -- VARIABLES REASONING` comment. The source shuold include lines up to (and including) the conditional/loop-logic definitions. DO NOT include the code block that the warp divergence points enclose, only the initial definition and necessary variables used in the warp divergence point entry logic.",
    )


class ConcretizationChecker(BaseModel):
    """ A class used to structure the output of the concretization checker node."""

    status: Literal["ACCEPT", "REJECT"] = Field(
        ..., 
        description="The status of the concretization check, which can be 'ACCEPT' or 'REJECT'. ACCEPT is given when the concretization follows all the rules and the source code is properly concretized. REJECT is given when the concretization does not follow the rules and the source code is not properly concretized."
        )
    rejectReason: str = Field(
        ..., 
        description="The brief reasoning behind a 'REJECT' status. Leave empty if the status is 'ACCEPT'."
        )
    

class SingleKernelState(BaseModel):
    """ A class used to structure the output of the single kernel source execution checker node."""

    status: Literal["ACCEPT", "REJECT"] = Field(
        ..., 
        description="The status of the single kernel source execution check, which can be 'ACCEPT' or 'REJECT'. ACCEPT is given when the source code modifications follow all the rules to transform the source code to have a single target kernel exeuction. REJECT is given when the modifications do not follow the rules and the source code is not properly transformed."
        )
    rejectReason: str = Field(
        ..., 
        description="The brief reasoning behind a 'REJECT' status. Leave empty if the status is 'ACCEPT'."
        )


# Updated KernelAnalysisState using TypedDict with default values for optional fields
class KernelAnalysisState(TypedDict, total=False):
    source_code: str
    kernel_name: str
    exec_args: str
    grid_size: str
    block_size: str
    total_num_threads: str

    # these will be filled in by the nodes

    # these are message logs used in retying the LLM calls with feedback
    # this add_messages ensures that when we return the state object dict, the messages are added to the messages list
    src_concretized_input_args: str
    step1_messages: Annotated[List, add_messages]
    concretizationState: ConcretizationChecker  


    src_single_kernel_execution: str
    step2_messages: Annotated[List, add_messages]
    srcSingleKernelState: SingleKernelState 

    snippet_first_kernel_invocation: str
    snippet_kernel_src: str
    snippet_kernel_src_concretized_values: str
    kernel_annotated_warp_divergence: str
    kernel_annotated_WDPs: str
    wdps_list: List[WarpDivergencePoint]  # List of tuples with warp divergence point source code and classification
    wdps_num_executions : List[int]

    kernel_annotated_num_ops: str
    summed_kernel_ops: str