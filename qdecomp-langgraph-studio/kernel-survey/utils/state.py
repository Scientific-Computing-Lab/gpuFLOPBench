from typing_extensions import TypedDict, List, Dict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from langgraph.graph.message import add_messages

class KernelAnalysisResult(BaseModel):
    has_float_division: bool
    float_division_explanation: str

    has_cuda_library_function_calls: bool
    cuda_library_function_calls_explanation: str

    has_recursion: bool
    recursion_explanation: str

    has_warp_divergence: bool
    warp_divergence_explanation: str

    has_data_dependent_warp_divergence: bool
    data_dependent_warp_divergence_explanation: str

    has_common_subexpression: bool
    common_subexpression_explanation: str

    has_special_math_function: bool
    special_math_function_explanation: str

class KernelAnalysisState(TypedDict, total=False):
    # dict mapping filenames to lists of kernel source codes
    source_codes: Dict[str, List[str]]
    results : Dict[str, Dict[str, KernelAnalysisResult]]
    target_name: str


class FloatDivCheck(BaseModel):
    has_float_division: bool = Field(..., description="Indicates whether the kernel source code contains floating-point division operations.")

    float_div_explanation: str = Field(..., description="One sentence explanation of the floating-point division operations found in the kernel source code. This should include the reasoning behind the presence of floating-point division operations.")

class CUDALibraryFunctionCallsCheck(BaseModel):
    has_cuda_library_function_calls: bool = Field(..., description="Indicates whether the kernel source code contains calls to CUDA library functions.")

    cuda_library_function_calls_explanation: str = Field(..., description="One sentence explanation of the CUDA library function calls found in the kernel source code. This should include the reasoning behind the presence of CUDA library function calls and their impact on the kernel's execution.")

class RecursionCheck(BaseModel):
    has_recursion: bool = Field(..., description="Indicates whether the kernel source code contains recursive function calls.")

    recursion_explanation: str = Field(..., description="One sentence explanation of the recursive function calls found in the kernel source code. This should include the reasoning behind the presence of recursion and its implications for the kernel's execution.")

class WarpDivergenceCheck(BaseModel):
    has_warp_divergence: bool = Field(..., description="Indicates whether the kernel source code contains warp divergence points. Warp divergence occurs when threads in a warp encounter different execution paths, such as an if statement, for-loop, switch, or while loop.")

    warp_divergence_explanation: str = Field(..., description="One sentence explanation of the warp divergence points found in the kernel source code. This should include the reasoning behind the presence of warp divergence and its implications for the kernel's execution.")

class DataDependentWarpDivergenceCheck(BaseModel):
    has_data_dependent_warp_divergence: bool = Field(..., description="Indicates whether the kernel source code contains warp divergence points whose logic are dependent on the value of data, either directly or indirectly (e.g: some sum variable is determined by the value of some data, and is used in a conditional). Warp divergence occurs when threads in a warp encounter different execution paths, such as an if statement, for-loop, switch, or while loop. We should indicate if any of the control flow logic of these warp divergence points are dependent on the value of data, either directly or indirectly.")

    data_dependent_warp_divergence_explanation: str = Field(..., description="One sentence explanation of the warp divergence points that are dependent on the value of data found in the kernel source code. This should include the reasoning as to why the warp divergence points are dependent on the value of data.")

class CommonSubexpressionEliminationCheck(BaseModel):
    has_common_subexpression: bool = Field(..., description="Indicates whether the kernel source code contains common subexpressions that would most likely be eliminated by the compiler.")

    common_subexpression_explanation: str = Field(..., description="One sentence explanation of the common subexpressions found in the kernel source code that would be eliminated by the compiler. This should include the reasoning behind the presence of such repeated subexpressions that would be eliminated by the compiler.")

class SpecialMathFunctionCheck(BaseModel):
    has_special_math_function: bool = Field(..., description="Indicates whether the kernel source code contains special math functions that are not standard arithmetic operations. This includes functions like sin, cos, exp, log, log2, sqrt, rsqrt, etc.")

    special_math_function_explanation: str = Field(..., description="One sentence explanation of the special math functions found in the kernel source code.") 

