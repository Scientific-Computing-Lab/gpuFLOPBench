from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# This is used for the structured output of the model
# the order of this class matters, as the model will first fill out the explanation, then the flop count
class FLOPCounts(BaseModel):
    sp_flop_explanation: str = Field(..., description="Explanation of how the single-precision floating point operations (SP-FLOP) count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")

    sp_flop_count: int = Field(..., description="Total number of single-precision floating point operations (SP-FLOP) performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")

    dp_flop_explanation: str = Field(..., description="Explanation of how the double-precision floating point operations (DP-FLOP) count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")

    dp_flop_count: int = Field(..., description="Total number of double-precision floating point operations (DP-FLOP) performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")


simpleSystemPrompt="""You are an expert CUDA source code FLOP counting assistant. For a given target CUDA kernel, you will be given: 
A) The Target Kernel Name
B) Amalgamated Source Code Files
C) Commandline Input Arguments
D) Grid and Block Size Launch parameters. 
Your task is to analyze the code and accurately determine the number of single-precision (SP-FLOP) and double-precision (DP-FLOP) floating point operations (FLOP) performed by the kernel during its FIRST execution invocation.

Provide a detailed explanation of how you arrived at the SP-FLOP and DP-FLOP counts, including any assumptions or simplifications you made during your analysis. Report the final SP-FLOP and DP-FLOP counts using the `sp_flop_count`, `dp_flop_count`, `sp_flop_explanation` and `dp_flop_explanation` fields in your response.
"""

complexSystemPrompt="""You are an expert CUDA source code FLOP counting assistant. For a given target CUDA kernel, you will be given: 
A) The Target Kernel Name
B) Amalgamated Source Code Files
C) Commandline Input Arguments
D) Grid and Block Size Launch parameters. 
Your task is to analyze the code and accurately determine the number of single-precision (SP-FLOP) and double-precision (DP-FLOP) floating point operations (FLOP) performed by the kernel during its FIRST execution invocation.

The steps you should generally follow are those of an expert human analyst (listed below):
Step 1) Propagate the input arguments and any constants through the source code
Step 2) Modify the source code to only keep the first execution of the target kernel
Step 3) Extract the target kernel code, relevant kernel input arguments and array sizes
Step 4) Propagate any input constants as well as grid and block size information into the kernel code
Step 5) Analyze the kernel code:
     5a) Identify the warp divergence points (WDPs) in the code (e.g: if-statements, for-loops, while-loops, switch statements)
     5b) For each WDP, determine the number of threads that will enter each branch of the WDP
     5c) For each branch of each WDP, count the number of SP-FLOP and DP-FLOP performed by a single thread in that branch
     5d) Multiply the SP-FLOP and DP-FLOP counts in each WDP branch by the number of threads that enter said branch
Step 6) Sum the SP-FLOP and DP-FLOP counts across all WDP branches to get the total SP-FLOP and DP-FLOP counts for the kernel
Step 7) Provide a detailed explanation of how you arrived at the SP-FLOP and DP-FLOP counts, including any assumptions or simplifications you made during your analysis. Report the final SP-FLOP and DP-FLOP counts using the `sp_flop_count`, `dp_flop_count`, `sp_flop_explanation` and `dp_flop_explanation` fields in your response.
"""


def make_prompt(prompt_type: str):

    sys_prompt = ""
    if prompt_type == "full":
        sys_prompt = complexSystemPrompt
    elif prompt_type == "simple":
        sys_prompt = simpleSystemPrompt
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    prompt = ChatPromptTemplate.from_messages([
         ("system", sys_prompt),
          ("human",
"""Target Kernel Name: {kernel_name}
Execution Arguments: {exec_args}
Grid Size: {grid_size}
Block Size: {block_size}
Total Number of Threads: {total_num_threads}
Please return the completed FLOP counts and explanations in the following fields: sp_flop_count, dp_flop_count, sp_flop_explanation, dp_flop_explanation. 
Source code:\n```{source_code}```
""")
    ])

    return prompt
