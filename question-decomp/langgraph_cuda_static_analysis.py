from typing import Annotated

from typing_extensions import TypedDict

from os import getenv
import os

import langgraph
from langgraph.graph import StateGraph, END
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any

import operator
from typing import Annotated

from IPython.display import Image, display


# please create a file called .openrouter-api-key in the current directory
with open('./.openrouter-api-key', 'r') as file:
    OPENROUTER_API_KEY=file.read().strip()
    #os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY

# setup our LLM
llm = ChatOpenAI(
  openai_api_key=OPENROUTER_API_KEY,
  openai_api_base="https://openrouter.ai/api/v1",
  #model_name="openai/o4-mini>",
  model_name="openai/gpt-4o-mini", # cheap model for testing
  model_kwargs={
    #"headers": {
      #"HTTP-Referer": getenv("YOUR_SITE_URL"),
      #"X-Title": getenv("YOUR_SITE_NAME"),
    #}
  },
)

def constant_reducer(a, b):
    # Always keep the first value; ignore others
    if a is None:
        return b
    return a

# Updated KernelAnalysisState using TypedDict with default values for optional fields
class KernelAnalysisState(TypedDict, total=False):
    source_code: str
    kernel_name: str
    exec_args: str
    grid_size: str
    block_size: str
    sass_imix: str
    sass_code: str

    # these will be filled in by the nodes
    updated_source_code: str
    kernel_op_estimates: str
    updated_sass_imix: str
    final_op_counts: str

# 1. Executable Input Args Parser and Replacer Node
def exec_args_parser_replacer(state: KernelAnalysisState, llm: ChatOpenAI):
    source_code = state["source_code"]
    kernel_name = state["kernel_name"]
    exec_args = state["exec_args"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code transformer that replaces all variable definitions and references in the given C/C++ CUDA source code with their corresponding hard-coded input argument literal values from the given execution arguments. "
         "If a value is derived from an input argument, also replace it with the hard-coded value. For CUDA kernel invocations, make sure all kernel input arguments are made explicit using the concrete values. "
         "If you cannot make a value concrete (e.g., pointers), leave it as-is. Only return the transformed source code, nothing else."),
        ("human", 
         "Source code:\n{source_code}\n\n"
         "Kernel name: {kernel_name}\n\n"
         "Execution arguments: {exec_args}\n\n"
         "Please return the updated source code with evaluated input arguments reflected in the kernel invocation call.")
    ])
    chain = prompt | llm
    updated_source = chain.invoke({
        "source_code": source_code,
        "kernel_name": kernel_name,
        "exec_args": exec_args,
    }).content

    print(f"Updated source code:\n{updated_source}\n")

    return {"updated_source_code": updated_source}
    #state["updated_source_code"] = updated_source
    #return state

# 2. Kernel Input Estimator Node
def kernel_input_estimator(state: KernelAnalysisState, llm: ChatOpenAI):
    updated_source = state["updated_source_code"]
    kernel_name = state["kernel_name"]
    grid_size = state["grid_size"]
    block_size = state["block_size"]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a static analysis assistant for CUDA kernels. Given the explicit C/C++ CUDA source code for a program and kernel, estimate the number of integer operations (INTOP), single-precision float operations (SP-FLOP), and double-precision float operations (DP-FLOP) performed by one CUDA kernel invocation. "
         "Report counts for FMA (fused-multiply-add), Add, Multiply, and Divide operations for each data type. If FMA is likely used (multiplication and addition on the same variables), count it as one op. "
         "Output strictly in this format:\n"
         "INTOP: XXX\nSP-FLOP ADD: XXX\nSP-FLOP MUL: XXX\nSP-FLOP DIV: XXX\nSP-FLOP FMA: XXX\nDP-FLOP ADD: XXX\nDP-FLOP MUL: XXX\nDP-FLOP DIV: XXX\nDP-FLOP FMA: XXX"
        ),
        ("human",
         "CUDA Source Code:\n{updated_source}\n\n"
         "Target Kernel name: {kernel_name}\nGrid size: {grid_size}\nBlock size: {block_size}\n\n"
         "Estimate the static instruction mix per target kernel invocation as described above.")
    ])
    chain = prompt | llm
    kernel_ops = chain.invoke({
        "updated_source": updated_source,
        "kernel_name": kernel_name,
        "grid_size": grid_size,
        "block_size": block_size,
    }).content

    print(f"Kernel operation estimates:\n{kernel_ops}\n")

    return {"kernel_op_estimates": kernel_ops}
    #state["kernel_op_estimates"] = kernel_ops
    #return state

# 3. SASS Dynamic IMIX Analyzer Node
def sass_dynamic_imix_analyzer(state: KernelAnalysisState, llm: ChatOpenAI):
    updated_source = state["updated_source_code"]
    sass_imix = state["sass_imix"]
    sass_code = state["sass_code"]
    grid_size = state["grid_size"]
    block_size = state["block_size"]
    kernel_name = state["kernel_name"]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a CUDA SASS instruction mix analyzer. Given the static SASS IMIX (instruction mix), SASS code for a kernel, and the corresponding C/C++ CUDA source code, return an updated SASS IMIX string containing the estimated dynamic execution counts for each instruction. "
         "Take into account the kernel input arguments, explicit loop bounds, and CUDA launch parameters (grid and block size). "
         "Consider jumps and loops in the SASS code that could repeat instructions. Output each instruction and its count as shown:\n"
         "BRA: 1\nF2I.FTZ.U32.TRUNC.NTZ: 1\nI2F.U32.RP: 1\nIADD3: 3\n... (one per line)"),
        ("human",
         "CUDA Source Code:\n{updated_source}\n\n"
         "Target Kernel Name: {kernel_name}\n\n"
         "SASS IMIX:\n{sass_imix}\n\n"
         "SASS Code:\n{sass_code}\n\n"
         "Grid size: {grid_size}\nBlock size: {block_size}\n\n"
         "Update the SASS IMIX to reflect dynamic execution counts per kernel launch as described.")
    ])
    chain = prompt | llm
    updated_sass_imix = chain.invoke({
        "updated_source": updated_source,
        "sass_imix": sass_imix,
        "sass_code": sass_code,
        "grid_size": grid_size,
        "block_size": block_size,
        "kernel_name": kernel_name,
    }).content

    print(f"Updated SASS IMIX:\n{updated_sass_imix}\n")

    return {"updated_sass_imix": updated_sass_imix}
    #state["updated_sass_imix"] = updated_sass_imix
    #return state

# 4. OPS Estimator Node
def ops_estimator(state: KernelAnalysisState, llm: ChatOpenAI):
    print("STATE TYPE")
    print(type(state))
    print("STATE TYPE")

    kernel_ops = state["kernel_op_estimates"]
    updated_sass_imix = state["updated_sass_imix"]
    updated_source = state["updated_source_code"]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an operation count estimator for CUDA kernels. Given the static instruction mix estimates (including FMA, ADD, MUL, DIV for INT, SP, DP) and the dynamic SASS IMIX, return an estimate of the total number of INTOP, SP-FLOP, and DP-FLOP performed by a single invocation of the kernel. "
         "Sum up the various ADD, MUL, and FMA ops for each type; FMA ops count as 2 operations each. Output strictly in this format:\n"
         "INTOP: XXX\nSP-FLOP: XXX\nDP-FLOP: XXX"),
        ("human",
         "Static instruction mix:\n{kernel_ops}\n\n"
         "Dynamic SASS IMIX:\n{updated_sass_imix}\n\n"
         "CUDA Source Code:\n{updated_source}\n\n"
         "Estimate total INTOP, SP-FLOP, DP-FLOP as described.")
    ])
    chain = prompt | llm
    final_ops = chain.invoke({
        "kernel_ops": kernel_ops,
        "updated_sass_imix": updated_sass_imix,
        "updated_source": updated_source,
    }).content

    print(f"Final operation counts:\n{final_ops}\n")

    return {"final_op_counts": final_ops}
    #state["final_op_counts"] = final_ops
    #return state

# Node wrappers for LangGraph
def make_exec_args_parser_node(llm):
    def node(state):
        return exec_args_parser_replacer(state, llm)
    return node

def make_kernel_input_estimator_node(llm):
    def node(state):
        return kernel_input_estimator(state, llm)
    return node

def make_sass_dynamic_imix_node(llm):
    def node(state):
        return sass_dynamic_imix_analyzer(state, llm)
    return node

def make_ops_estimator_node(llm):
    def node(state):
        return ops_estimator(state, llm)
    return node

# Build the graph
def build_cuda_kernel_ops_graph(llm, show_mermaid_png: bool = False):
    workflow = StateGraph(KernelAnalysisState)

    workflow.add_node("exec_args_parser", make_exec_args_parser_node(llm))
    workflow.add_node("kernel_input_estimator", make_kernel_input_estimator_node(llm))
    workflow.add_node("sass_dynamic_imix", make_sass_dynamic_imix_node(llm))
    workflow.add_node("ops_estimator", make_ops_estimator_node(llm))

    # Graph edges
    #workflow.add_edge("exec_args_parser", ["kernel_input_estimator", "sass_dynamic_imix", "ops_estimator"])
    workflow.add_edge("exec_args_parser", "kernel_input_estimator")
    workflow.add_edge("exec_args_parser", "sass_dynamic_imix")
    workflow.add_edge(["exec_args_parser", "kernel_input_estimator", "sass_dynamic_imix"], "ops_estimator")
#    workflow.add_edge("exec_args_parser", "ops_estimator")
#    workflow.add_edge("kernel_input_estimator", "ops_estimator")
#    workflow.add_edge("sass_dynamic_imix", "ops_estimator")
    workflow.add_edge("ops_estimator", END)

    # Set entrypoint
    workflow.set_entry_point("exec_args_parser")
    compiled = workflow.compile()

    # Draw and save the graph as a PNG image if requested
    if show_mermaid_png:
        display(Image(compiled.get_graph().draw_mermaid_png()))

    return compiled

# -- Example usage for initializing the state from user data --


# Usage Example:
graph = build_cuda_kernel_ops_graph(llm, show_mermaid_png=True)
#result = graph.invoke(query_data)

#print(result['final_op_counts'])
