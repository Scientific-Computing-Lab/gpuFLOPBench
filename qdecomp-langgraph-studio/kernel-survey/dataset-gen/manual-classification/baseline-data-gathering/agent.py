# make a langchain graph instance with the model
from typing_extensions import TypedDict, List, Annotated, Literal
from pydantic import BaseModel, Field
from dataset_and_llm import llm
from langchain.prompts import ChatPromptTemplate

class FLOPCounts(BaseModel):
    sp_flop_count: int = Field(..., description="Total number of single-precision floating point operations (SP-FLOP) performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")

    sp_flop_explanation: str = Field(..., description="Explanation of how the single-precision floating point operations (SP-FLOP) count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")

    dp_flop_count: int = Field(..., description="Total number of double-precision floating point operations (DP-FLOP) performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")

    dp_flop_explanation: str = Field(..., description="Explanation of how the double-precision floating point operations (DP-FLOP) count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")

class BaselineQueryState(TypedDict, total=False):
    source_code: str
    combined_name: str
    kernel_name: str
    exec_args: str
    grid_size: str
    block_size: str
    total_num_threads: str
    empirical_sp_flop_count: float
    empirical_dp_flop_count: float

    flop_counts: FLOPCounts

    predicted_sp_flop_count: int
    predicted_dp_flop_count: int

# Calculate the total number of threads from the gridSz and the blockSz
# grid size is a string of format "(x, y, z)"
# block size is a string of format "(x, y, z)"
def calc_total_threads(gridSz:str, blockSz:str):
    gridSz = eval(gridSz)
    blockSz = eval(blockSz)
    total_threads = gridSz[0] * gridSz[1] * gridSz[2] * blockSz[0] * blockSz[1] * blockSz[2]
    return str(total_threads)

def get_input_problem(state: BaselineQueryState, config):
    verbose = config.get("configurable", {}).get("verbose_printing", False)

    row = config.get("configurable", {}).get("input_problem_row", None) 

    target_name = row['target_name']

    assert row is not None, f"Target problem '{target_name}' not found in the dataset."

    #print(exeArgs_list)
    if verbose:
        print("---------- BEGIN STEP 0: GET INPUT PROBLEM ----------", flush=True)

    to_return = {'source_code' : row['source_code'], 
            'combined_name' : row['combined_name'],
            'kernel_name' : row['Kernel Name'],
            'exec_args' : row['exeArgs'],
            'grid_size' : row['Grid Size'],
            'block_size' : row['Block Size'],
            'total_num_threads' : calc_total_threads(row['Grid Size'], row['Block Size']),
            # these "true" values do not get passed to the LLMs
            # they are used to calculate how close the LLM prediction is to the ground-truth
            'empirical_sp_flop_count' : row['SP_FLOP'],
            'empirical_dp_flop_count' : row['DP_FLOP'],
            }

    if verbose:
        for k, v in to_return.items():
            if k != "source_code":
                print(f"\t{k}: {v}", flush=True)
        print("---------- END STEP 0: GET INPUT PROBLEM ----------", flush=True)

    return to_return


def query_for_flop_count(state: BaselineQueryState, config):
    verbose = config.get("configurable", {}).get("verbose_printing", False)

    configured_llm = llm.with_config(configurable=config.get("configurable", {})).with_structured_output(FLOPCounts, include_raw=True)

    # this node is used to check how well the concretization worked
    prompt = ChatPromptTemplate.from_messages([
         ("system",
         """
         You are an expert CUDA source code FLOP counting assistant. You will be given a CUDA kernel's source code and its execution configuration. Your task is to analyze the code and accurately determine the number of single-precision (SP) and double-precision (DP) floating point operations (FLOPs) performed by the kernel during its execution.
"""
          ),
          ("human",
           """Hello world
""")
    ])

    chain = prompt | concretization_checker_llm

    result = chain.invoke({
        "concretization_rules": concretization_rules,
    })

    resultState = result['parsed']

    if verbose:
        print(f"\tSTEP (1) Concretization Checker Result: [{resultState.status}]", flush=True)

    updated_costs = get_query_cost(result['raw'], verbose)

    return updated_costs | {"concretizationState": resultState}