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
  temperature=0.2,
  top_p=0.1,
  #model_name="openai/o4-mini>",
  model_name="openai/gpt-4o-mini", # cheap model for testing
  #model_kwargs={
  #  'top_p' : 0.1,
  #},
)


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

example_before = """
Example Executable Input Arguments: [664]
---- EXAMPLE BEFORE Transformation ----
#include <stdio.h>
#include <cstdlib>

// templated function
template <std::size_t MYNUM, typename T>
int add(T a, T b) {
    return a + b*((int)MYNUM);
}

#define N 1024
int main(int argc, char** argv) {
    int M = (argc == 2) ? atoi(argv[1]) : 2;
    int *data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        data[i] = add<3, int>(i, M);
    }
    printf("Data initialized.\\n");
}
"""

example_after = """
Example Executable Input Arguments: [664]
---- EXAMPLE AFTER Transformation ----
#include <stdio.h>
#include <cstdlib>

// templated function
template <std::size_t MYNUM, typename T>
int add(T a, T b) {
    // values passed in as template parameters or function arguments are replaced with their concrete values
    return a + 664*((int)3);
}

#define N 1024
int main(int argc, char** argv) {
    int M = (argc == 2) ? atoi(argv[1]) : 2;
    int *data = (int*)malloc(1024 * sizeof(int));
    // loop bounds and preprocessor defines are also replaced with concrete values
    for (int i = 0; i < 1024; i++) {
        data[i] = add<3, int>(i, 664);
    }
    printf("Data initialized.\\n");
}
"""

# 1. Executable Input Args Parser and Replacer Node
def exec_args_parser_replacer(state: KernelAnalysisState, llm: ChatOpenAI):
    source_code = state["source_code"]
    kernel_name = state["kernel_name"]
    exec_args = state["exec_args"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code transformer that replaces all variable definitions, preprocessor defines, template parameters, and references in the given C/C++ CUDA source code with their corresponding hard-coded input argument literal values from the given execution arguments and evaluated source code."
         "If a value is derived from another value, also replace it with the hard-coded value. For CUDA kernel invocations, make sure all possible kernel input arguments are made explicit using the concrete values. "
         "If you cannot make a value concrete (e.g.: pointers), leave it as-is. Only return the transformed source code, nothing else.\n"
         "Below is an example of the desired types of variable and explicit value concretization source code transformations:\n"
         "```{example_before}```\n\n"
         "```{example_after}```\n\n"),
        ("human", 
         "Target Kernel name: {kernel_name}\n\n"
         "Execution arguments: {exec_args}\n\n"
         "Please return the updated source code with evaluated input arguments, variables, references, and preprocessor defines. Ensure to reaplce as many variables as possible with their literal values in the kernel invocation call."
         "Source code:\n{source_code}\n\n")
    ])
    chain = prompt | llm
    updated_source = chain.invoke({
        "source_code": source_code,
        "kernel_name": kernel_name,
        "exec_args": exec_args,
        "example_before": example_before,
        "example_after": example_after,
    }).content

    print(f"Updated source code:\n{updated_source}\n")

    return {"updated_source_code": updated_source}
    #state["updated_source_code"] = updated_source
    #return state

# 2. Static Instruction Mix (IMIX) Estimator Node
def static_imix_estimator(state: KernelAnalysisState, llm: ChatOpenAI):
    updated_source = state["updated_source_code"]
    kernel_name = state["kernel_name"]
    grid_size = state["grid_size"]
    block_size = state["block_size"]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a static analysis assistant for CUDA kernels. Given the explicit C/C++ CUDA source code for a program and kernel, estimate the number of integer operations (INTOP), single-precision float operations (SP-FLOP), and double-precision float operations (DP-FLOP) performed by one CUDA kernel invocation. "
         "Report counts for FMA (fused-multiply-add), Add (ADD), Multiply (MUL), and Divide (DIV) operations for each data type. If FMA is likely to be used by the compiler (multiplication and addition on the same variables), count it as one op. "
         "Output strictly in this format:\n"
         "INTOP ADD: XXX\nINTOP MUL: XXX\nINTOP DIV: XXX\nINTOP FMA: XXX\nSP-FLOP ADD: XXX\nSP-FLOP MUL: XXX\nSP-FLOP DIV: XXX\nSP-FLOP FMA: XXX\nDP-FLOP ADD: XXX\nDP-FLOP MUL: XXX\nDP-FLOP DIV: XXX\nDP-FLOP FMA: XXX"
        ),
        ("human",
         "Target Kernel name: {kernel_name}\nGrid size: {grid_size}\nBlock size: {block_size}\n\n"
         "Estimate the static instruction mix per target kernel invocation as described above."
         "CUDA Source Code:\n{updated_source}\n\n")
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
         "Take into account the kernel input arguments, explicit loop bounds, conditional branching / warp divergence, and CUDA launch parameters (grid and block size). "
         "Consider jumps and loops in the SASS code that could repeat instructions. Output each instruction and its count as shown:\n"
         "BRA: XXX\nF2I.FTZ.U32.TRUNC.NTZ: XXX\nI2F.U32.RP: XXX\nIADD3: XXX\n... (one per line)"),
        ("human",
         "Target Kernel Name: {kernel_name}\n\n"
         "SASS IMIX:\n{sass_imix}\n\n"
         "Grid size: {grid_size}\nBlock size: {block_size}\n\n"
         "Update the SASS IMIX to reflect dynamic execution counts per kernel launch as described."
         "CUDA Source Code:\n{updated_source}\n\n\n"
         "SASS Code:\n{sass_code}\n\n")
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
         "You are an operation count estimator for CUDA kernels. Given the static instruction mix estimates (including FMA, ADD, MUL, DIV for INT, SP, DP) and the dynamic SASS IMIX estimates, return an estimate of the total number of INTOP, SP-FLOP, and DP-FLOP performed by a single invocation of the kernel. \n"
         "Recall that each "
         "Sum up the various ADD, MUL, and FMA ops for each operation type; FMA ops count as 2 operations each. Output strictly in this format:\n"
         "INTOP: XXX\nSP-FLOP: XXX\nDP-FLOP: XXX"),
        ("human",
         "Static instruction mix:\n{kernel_ops}\n\n"
         "Dynamic SASS IMIX:\n{updated_sass_imix}\n\n"
         "Estimate total INTOP, SP-FLOP, DP-FLOP as described."
         "CUDA Source Code:\n{updated_source}\n\n")
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

def make_static_imix_estimator_node(llm):
    def node(state):
        return static_imix_estimator(state, llm)
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
    workflow.add_node("static_imix_estimator", make_static_imix_estimator_node(llm))
    workflow.add_node("sass_dynamic_imix", make_sass_dynamic_imix_node(llm))
    workflow.add_node("ops_estimator", make_ops_estimator_node(llm))

    # Graph edges
    #workflow.add_edge("exec_args_parser", ["static_imix_estimator", "sass_dynamic_imix", "ops_estimator"])
    workflow.add_edge("exec_args_parser", "static_imix_estimator")
    workflow.add_edge("exec_args_parser", "sass_dynamic_imix")
    workflow.add_edge(["exec_args_parser", "static_imix_estimator", "sass_dynamic_imix"], "ops_estimator")
#    workflow.add_edge("exec_args_parser", "ops_estimator")
#    workflow.add_edge("static_imix_estimator", "ops_estimator")
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
