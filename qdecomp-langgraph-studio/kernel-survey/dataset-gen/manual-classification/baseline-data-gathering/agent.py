# make a langchain graph instance with the model
from typing_extensions import TypedDict, List, Annotated, Literal
from dataset_and_llm import llm
from prompts import make_prompt, FLOPCounts
from io_cost import get_query_cost
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.schema import AIMessage
from configuration import Configuration
import sqlite3

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

    prompt_type: Literal["simple", "full"]

    raw_flop_counts: AIMessage

    predicted_sp_flop_count: int
    predicted_dp_flop_count: int
    predicted_sp_flop_count_explanation: str
    predicted_dp_flop_count_explanation: str
    
    input_tokens: Annotated[List[int], operator.add]
    output_tokens: Annotated[List[int], operator.add]
    total_cost: Annotated[List[float], operator.add]

    total_query_time: float
    error: str

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

    prompt_type = config.get("configurable", {}).get("prompt_type", "simple")

    combined_name = row['combined_name']

    assert row is not None, f"Target problem '{combined_name}' not found in the dataset."

    if verbose:
        print("---------- BEGIN STEP 0: GET INPUT PROBLEM ----------", flush=True)

    to_return = {'source_code' : row['source_code'], 
            'combined_name' : combined_name,
            'kernel_name' : row['Kernel Name'],
            'exec_args' : row['exeArgs'],
            'grid_size' : row['Grid Size'],
            'block_size' : row['Block Size'],
            'total_num_threads' : calc_total_threads(row['Grid Size'], row['Block Size']),
            # these "true" values do not get passed to the LLMs
            # they are used to calculate how close the LLM prediction is to the ground-truth
            'empirical_sp_flop_count' : row['SP_FLOP'],
            'empirical_dp_flop_count' : row['DP_FLOP'],
            'prompt_type' : prompt_type
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

    prompt = make_prompt(state['prompt_type'])

    chain = prompt | configured_llm 

    if verbose:
        print("---------- BEGIN STEP 1: QUERY FOR FLOP COUNT ----------", flush=True)
        print(f"\tQuerying for FLOP count of kernel: {state['combined_name']}", flush=True)

    result = chain.invoke({
        "source_code": state['source_code'],
        "kernel_name": state['kernel_name'],
        "exec_args": state['exec_args'],
        "grid_size": state['grid_size'],
        "block_size": state['block_size'],
        "total_num_threads": state['total_num_threads']
    })

    parsed_result = result['parsed']

    if verbose:
        print(f"\tGot an LLM response!: \n\tSP_FLOP:[{parsed_result.sp_flop_count}], \n\tDP_FLOP:[{parsed_result.dp_flop_count}]\n", flush=True)
        result['raw'].pretty_print()

    query_cost = get_query_cost(result['raw'], verbose)

    return query_cost | {'predicted_sp_flop_count': parsed_result.sp_flop_count, 
                         'predicted_dp_flop_count': parsed_result.dp_flop_count, 
                         'predicted_sp_flop_count_explanation': parsed_result.sp_flop_explanation, 
                         'predicted_dp_flop_count_explanation': parsed_result.dp_flop_explanation, 
                         'raw_flop_counts': result['raw']
                        }


def make_graph(sqlite_db_path: str):
    # now let's set up the StateGraph to represent the agent
    workflow = StateGraph(BaselineQueryState, context_schema=Configuration)
    workflow.add_node("get_input_problem_0", get_input_problem)
    workflow.add_node("query_for_flop_count_1", query_for_flop_count)

    workflow.add_edge("get_input_problem_0", "query_for_flop_count_1")
    workflow.add_edge("query_for_flop_count_1", END)

    workflow.set_entry_point("get_input_problem_0")

    # let's also add a checkpointer to save intermediate results
    # sqlite_db_path: path to sqlite database used by SqliteSaver to persist graph checkpoints
    conn = sqlite3.connect(sqlite_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = workflow.compile(checkpointer=checkpointer)

    return graph



