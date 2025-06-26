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
    acceptReason: str = Field(
        ..., 
        description="The brief reasoning behind a 'ACCEPT' status. Leave empty if the status is 'REJECT'."
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
    acceptReason: str = Field(
        ..., 
        description="The brief reasoning behind a 'ACCEPT' status. Leave empty if the status is 'REJECT'."
        )

class NumOpsState(BaseModel):
    """ A class used to structure the output of the num ops annotator checker node."""

    status: Literal["ACCEPT", "REJECT"] = Field(
        ..., 
        description="The status of the FLOP operations count checker, which can be 'ACCEPT' or 'REJECT'. ACCEPT is given when the source code modifications correctly follow all the rules to annotate the source code with comments correctly indicating the number of single-precision (SP-FLOP) and double-precision (DP-FLOP) floating point operations on the lines where they occur. REJECT is given when the modifications do not follow the rules and the source code is NOT properly annotated with the correct SP-FLOP or DP-FLOP counts, or the source code has annotations that are incorrect/inaccurate."
        )
    rejectReason: str = Field(
        ..., 
        description="The brief reasoning behind a 'REJECT' status. Leave empty if the status is 'ACCEPT'."
        )
    acceptReason: str = Field(
        ..., 
        description="The brief reasoning behind a 'ACCEPT' status. Leave empty if the status is 'REJECT'."
        )

# We create a custom structured output so that models return the number of iterations executed for each warp divergence point
class NumExecutions(BaseModel):
    num_executions: int = Field(..., description="Calculated number of times the source code will be executed based on the mathematical summation logic provided in the prompt. This is a single integer value representing the total number of times the given code snippet will be executed for the provided conditional values. -1 indicates that we are unable to calculate an exact integer number of executions.")

    num_executions_explanation: str = Field(..., description="Explanation of how the number of executions was calculated. This should include the reasoning behind the number of executions performed. The explanation should be clear and concise, providing insight into the mathematical logic used to derive the number of executions.")


class FLOPCounts(BaseModel):
    sp_flop_count: int = Field(..., description="Total number of single-precision floating point operations (SP-FLOP) performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")

    sp_flop_explanation: str = Field(..., description="Explanation of how the single-precision floating point operations (SP-FLOP) count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")

    dp_flop_count: int = Field(..., description="Total number of double-precision floating point operations (DP-FLOP) performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")

    dp_flop_explanation: str = Field(..., description="Explanation of how the double-precision floating point operations (DP-FLOP) count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")


# Updated KernelAnalysisState using TypedDict with default values for optional fields
class KernelAnalysisState(TypedDict, total=False):
    source_code: str
    kernel_name: str
    exec_args: str
    grid_size: str
    block_size: str
    total_num_threads: str
    empirical_sp_flop_count: float
    empirical_dp_flop_count: float
    treesitter_propagated_source_code: str

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
    step5_messages: Annotated[List, add_messages]
    snippetConcretizationState: ConcretizationChecker

    kernel_annotated_warp_divergence: str

    # Source snippet annotated with the warp divergence points and their dependent variable value ranges
    kernel_annotated_WDPs: str

    wdps_list: List[WarpDivergencePoint]  # List of tuples with warp divergence point source code and classification
    wdps_num_executions: List[NumExecutions] # Corresponding number of executions for each warp divergence point

    kernel_annotated_num_ops: str
    step8_messages: Annotated[List, add_messages]
    numOpsAnnotationState: NumOpsState

    summed_kernel_ops: FLOPCounts

    sp_flop_diff: float
    dp_flop_diff: float

    sp_flop_perc_diff: float
    dp_flop_perc_diff: float