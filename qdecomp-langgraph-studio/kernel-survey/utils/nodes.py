from utils.state import KernelAnalysisState, FloatDivCheck, CUDALibraryFunctionCallsCheck, RecursionCheck, WarpDivergenceCheck, DataDependentWarpDivergenceCheck, CommonSubexpressionEliminationCheck, SpecialMathFunctionCheck 

from typing_extensions import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

#from langchain.chat_models import init_chat_model
from langchain_core.runnables import ConfigurableField

from .dataset import kernels_data, target_names

from utils.preprocessing.args_propagator import *
from utils.preprocessing.source_reorganizer import *
from utils.preprocessing.prune_irrelevant_code import *

import os

# The ids of the configurables are from the Configuration class in configuration.py
# This is needed to allow us to change the variables at runtime
llm = ChatOpenAI(
  openai_api_key="",
  openai_api_base="https://openrouter.ai/api/v1",
  temperature=0.2,
  top_p=0.1,
  model_name="openai/o3-mini",
).configurable_fields(
    model_name=ConfigurableField(
        id="model",
    ),
    temperature=ConfigurableField(
        id="temp",
    ),
    top_p=ConfigurableField(
        id="top_p",
    ),
    openai_api_base=ConfigurableField(
        id="provider_url",
    ),
    openai_api_key=ConfigurableField( 
        id="provider_api_key",
    )
)



def get_input_problem(state: KernelAnalysisState, config):

    target_name = config.get("configurable", {}).get("input_problem", "resize-cuda")

    return {
            'source_codes' : target_names[target_name],
            'target_name' : target_name,
            }



def float_div_check_node(state: KernelAnalysisState, config):

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code transformer that replaces all variable definitions, preprocessor defines, template parameters, and references in the given C/C++ CUDA source code with their corresponding hard-coded input argument literal values from the given execution arguments and evaluated/derived source code values.\n"
         "Be sure to follow the following RULES when transforming the source code:\n{concretization_rules}\n"
         "Below is an example of the desired types of variable and explicit value concretization source code transformations:\n"
         "{step1_example_before}\n\n"
         "{step1_example_after}\n\n"
         ),
        ("human", 
         "Target Kernel Name: {kernel_name}\n"
         "Execution Arguments: {exec_args}\n"
         "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
         "Please return the updated source code with evaluated input arguments, variables, references, template arguments, and preprocessor defines. Ensure to calculate as many variables as possible with their literal values in the target kernel invocation call and any intermediate variables that get calculated."
         "Source code:\n```{source_code}```\n\n"
         )
    ])
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))

    inputs = {
        "source_code": state["source_code"],
        "kernel_name": state["kernel_name"],
        "exec_args": state["exec_args"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"], 
        "total_num_threads": state["total_num_threads"], 
        "step1_example_before": step1_example_before,
        "step1_example_after": step1_example_after,
        "concretization_rules": concretization_rules,
    }

    result = chain.invoke(inputs)

    updated_source = result.content

    #print("\n\n\n")
    #print("---------- BEGIN STEP 1: Source Code Concretization ----------")
    #print(f"\n{updated_source}\n")
    #print("---------- END STEP 1: Source Code Concretization ----------")
    #print("\n\n\n")

    return {"src_concretized_input_args": updated_source,
                "step1_messages": prompt.format_messages(**inputs) + [result]}





# we're going to force this node to give us structured output (i.e: a tool call)
def concretization_checker(state: KernelAnalysisState, config):

    concretization_checker_llm = llm.with_config(configurable=config.get("configurable", {})).with_structured_output(ConcretizationChecker)

    msg_histroy = state["step1_messages"]

    # this node is used to check how well the concretization worked
    prompt = ChatPromptTemplate.from_messages([
         ("system",
          "You are a code checker that verifies the concretization of the given C/C++ CUDA source code.\n" 
          "Make sure that the returned concretized code follows the rules of the original system message.\n"
          "If the concretization follows all the rules, it is correct, and you should return the ACCEPT status tool call.\n"
          "The acceptReason field should explain why the 'ACCEPT' status was given, and state all the reasons why the concretization is correct.\n"
          "If the concretization fails to follow at least one rule, it is incorrect, and you should return the REJECT status tool call with a brief rejectReason explaining why the concretization is incorrect or what it may be missing.\n"
          "The rejectReason should state all the possible reasons why the concretization may be incorrect."
          "Be sure to check that the produced code follows ALL the rules of the original system message. Failure to do so should result in a REJECT status.\n"
          "Below are the rules that the concretization must follow:\n"
          "{concretization_rules}\n"
          "Original system message and concretized source code response messages are provided below.\n"
          ),
    ] + msg_histroy)

    chain = prompt | concretization_checker_llm

    resultState = chain.invoke({
        "concretization_rules": concretization_rules,
    })

    return {"concretizationState": resultState}


def route_concretization_status_edge(state: KernelAnalysisState):
    """
    This function routes the edge based on the concretization status.
    If the concretization is good, it returns the next node to execute.
    If the concretization is not good, it returns the get_input_problem node to retry.
    """
    return state.get("concretizationState", {}).status










with open('./example_codes/step2_example_before.cu', 'r') as file:
    step2_example_before = file.read()

with open('./example_codes/step2_example_after.cu', 'r') as file:
    step2_example_after = file.read()


def src_single_kernel_execution_modifier(state: KernelAnalysisState, config): 

    if len(state["step2_messages"]) != 0:
        msg_history = state["step2_messages"]
        singleKernelState = state["srcSingleKernelState"]

        error_msg = ChatPromptTemplate.from_messages(msg_history + [
             ("assistant",
              "The single kernel source transformation was rejected with the following reason(s):\n{reason}\n"
              "Please update the erroneous transformed source code with the necessary changes to make the concretization correct.\n")
        ])

        chain = error_msg | llm.with_config(configurable=config.get("configurable", {}))

        inputs = {
            "reason": singleKernelState.rejectReason,
        }

        result = chain.invoke(inputs)

        updated_source = result.content

        return {"src_single_kernel_execution": updated_source,
                "step2_messages": error_msg.format_messages(**inputs) + [result]}
    
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a code transformer that modifies the given C/C++ CUDA source code to ensure that only a single kernel invocation of the target kernel name is executed.\n"
             "This means removing any loops or multiple invocations of the kernel, to leave only the first invocation of the target kernel in the code. This also means ensuring that the kernel is invoked with the correct arguments, grid, and block sizes.\n"
             "The modifications should be done by commenting out parts of the original code to be changed, and adding the changes on a new line below the original commented code.\n"
             "If an entire function ceases to be used or called due to commenting, omit that function from the transformed source code.\n"
             "If more than one CUDA kernel appears in the source code, and if the target CUDA kernel does not depend on said other kernels, remove the other kernels and their invocations from the source code.\n"
             "Only return the modified source code, nothing else.\nAn example is provided below:\n"
             "Example Before:\n"
             "{step2_example_before}\n\n"
             "Example After:\n"
             "{step2_example_after}\n\n"),
            ("human", 
             "Target Kernel Name: {kernel_name}\n"
             "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
             "Please return the updated source code with only a single kernel invocation."
             "Source code:\n{updated_source}\n"
             )
        ])
        chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
        inputs = {
            "updated_source": state["src_concretized_input_args"],
            "kernel_name": state["kernel_name"],
            "grid_size": state["grid_size"],
            "block_size": state["block_size"], 
            "total_num_threads": state["total_num_threads"], 
            "step2_example_before": step2_example_before,
            "step2_example_after": step2_example_after,
        }

        result = chain.invoke(inputs)

        single_kernel_source = result.content

        #print("\n\n\n")
        #print("---------- BEGIN STEP 2: Single Kernel Execution Modification ----------")
        #print(f"\n{single_kernel_source}\n")
        #print("---------- END STEP 2: Single Kernel Execution Modification ----------")
        #print("\n\n\n")

        return {"src_single_kernel_execution": single_kernel_source, 
                "step2_messages": prompt.format_messages(**inputs) + [result]}





def single_kernel_execution_checker(state: KernelAnalysisState, config):

    single_source_checker_llm = llm.with_config(configurable=config.get("configurable", {})).with_structured_output(SingleKernelState)

    msg_histroy = state["step2_messages"]

    # this node is used to check how well the single kernel source modifications worked
    prompt = ChatPromptTemplate.from_messages([
         ("system",
          "You are a code checker that verifies the correct modification of the given C/C++ CUDA source code.\n" 
          "The resulting source code should only contain a single invocation of the target kernel name, with the correct grid and block sizes, and no other kernels or loops that invoke the target kernel multiple times.\n"
          "Make sure that the returned source code follows the rules of the original system message.\n"
          "If the transformed code follows all the rules, it is correct, and you should return the ACCEPT status tool call.\n"
          "The acceptReason field should explain why the 'ACCEPT' status was given, and state all the possible reasons why the transformation is correct.\n"
          "If the transformed code fails to follow at least one rule, it is incorrect, and you should return the REJECT status tool call with a brief rejectReason explaining why the transformation is incorrect or what it may be missing.\n"
          "Be sure to check that the produced code follows ALL the rules of the original system message. Failure to do so should result in a REJECT status.\n"
          "Original system message and transformed source code response messages are provided below.\n"
          ),
    ] + msg_histroy)

    chain = prompt | single_source_checker_llm 

    resultState = chain.invoke({})

    return {"srcSingleKernelState": resultState}


def route_single_kernel_source_status_edge(state: KernelAnalysisState):
    return state.get("srcSingleKernelState", {}).status















def first_kernel_invocation_snippet_extractor(state: KernelAnalysisState, config):

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code scraper that extracts the first kernel invocation snippet from the given C/C++ CUDA source code. "
         "This means identifying the first kernel invocation/call and returning only that line(s) of the code, including the kernel name, grid size, block size, any necessary concrete input parameters, and template inputs."
         "Ensure that any concrete values are included in the invocation, such as the grid and block sizes, and any other input arguments that are passed to the kernel."
         "If the kernel invocation is made with pointers, structs, or objeects, include the variable names and a comment describing the datatype, size, and layout of the data they point to."
         "Only return the extracted CUDA kernel invocation snippet and related input descriptions, nothing else.\n"
         "Here is an example of the output format to return:"
         """// Template arguments for the exampleKernel invocation:
// DataType (first template argument, instantiated as 'float' for this call):
//          Specifies the data type of the elements in the input and output arrays.
//          For this call, it's 'float'.
// KERNEL_STENCIL_SIZE (second template argument, instantiated as '3' for this call):
//          An integer constant specifying the number of random neighbors (from the 3x3 neighborhood,
//          including the element itself as a possibility) to select and average.
//          For this call, it's '3'.

// Parameter descriptions for the exampleKernel<DataType, KERNEL_STENCIL_SIZE> invocation:
// d_input: DataType* (resolved to float* for this call), pointer to the input 2D array data on the GPU.
//          The data is a contiguous block of 'arraySize' (width * height) elements of type DataType.
//          Layout: Flattened 1D array in row-major order, logically representing a 'width' x 'height' 2D grid.
//          Size on GPU: arraySize * sizeof(DataType) bytes (e.g., 64 * sizeof(float) bytes).
// d_output: DataType* (resolved to float* for this call), pointer to the output 2D array data on the GPU.
//           The data is a contiguous block of 'arraySize' (width * height) elements of type DataType.
//           Layout: Flattened 1D array in row-major order, logically representing a 'width' x 'height' 2D grid.
//           Size on GPU: arraySize * sizeof(DataType) bytes (e.g., 64 * sizeof(float) bytes).
// width: int, the width (number of columns) of the logical 2D grid being processed.
//        Value for this call: 8.
// height: int, the height (number of rows) of the logical 2D grid being processed.
//         Value for this call: 8.
// seed: unsigned int, the seed value used to initialize pseudo-random number
//       generators within each CUDA thread for the stencil operation.

// Description for kernel launch configuration:
// Grid Size: 128 * 512 = (65536, 1, 1), the number of blocks in the grid for the kernel launch.
// Block Size: 32 * 32 = (1024, 1, 1), the number of threads in each block for the kernel launch.
// Total Number of Threads: 65536 * 1 * 1 * 1024 * 1 * 1 = 67108864 threads

exampleKernel<DataType=float, KERNEL_STENCIL_SIZE=3><<<gridSize=(65536, 1, 1), blockSize=(1024, 1, 1)>>>(d_input, d_output, width=8, height=8, seed);
"""
         ),
        ("human",
            "Target CUDA Kernel Name: ```{kernel_name}```\n"
            "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
            "Please return the snippet of the source code that contains the first kernel invocation with concrete values and associated input descriptions."
            "Source code:\n{updated_source}\n"
            )
    ])
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
    first_kernel_invocation = chain.invoke({
        "updated_source": state["src_single_kernel_execution"],
        "kernel_name": state["kernel_name"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"], 
        "total_num_threads": state["total_num_threads"], 
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 3: First Kernel Invocation Snippet Extraction ----------")
    print(f"\n{first_kernel_invocation}\n")
    print("---------- END STEP 3: First Kernel Invocation Snippet Extraction ----------")
    print("\n\n\n")

    return {"snippet_first_kernel_invocation": first_kernel_invocation}









def kernel_source_snippet_extractor(state: KernelAnalysisState, config):

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code scraper that extracts the source code snippet of the target kernel from the given C/C++ CUDA source code. "
         "This means identifying the target CUDA kernel function definition and returning only that part of the code, including the function signature, body, and any other __device__ or __global__ CUDA kernels the target CUDA kernel may call within its body."
         "Only return the extracted kernel source code and associated GPU kernel snippets, nothing else.\n"
         "There may exist multuple CUDA kernels in the source code, but you should only return the one that matches the target kernel name and any kernels which the target kernel makes calls to.\n"
         ),
        ("human",
            "Target CUDA Kernel Name: ```{kernel_name}```\n"
            "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
            "Please return the snippet of the source code that contains the kernel function definition."
            "Source code:\n{updated_source}\n"
            )
    ])
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
    kernel_source_snippet = chain.invoke({
        "updated_source": state["src_single_kernel_execution"],
        "kernel_name": state["kernel_name"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"], 
        "total_num_threads": state["total_num_threads"], 
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 4: Kernel Source Snippet Extraction ----------")
    print(f"\n{kernel_source_snippet}\n")
    print("---------- END STEP 4: Kernel Source Snippet Extraction ----------")
    print("\n\n\n")

    return {"snippet_kernel_src": kernel_source_snippet}









with open('./example_codes/step5_example_before.cu', 'r') as file:
    step5_example_before = file.read()

with open('./example_codes/step5_example_after.cu', 'r') as file:
    step5_example_after = file.read()

snippet_concretization_rules = """1) If a value is derived from other value(s), also replace it with the hard-coded value. 
2) Make sure all possible variables and arguments are made explicit using the provided hard-coded values and their descriptions.
3) If the `auto` keyword is used, replace it with the correct concrete type.
4) Be sure to replace any blockDim and gridDim variables (e.g: `blockDim.x` or `gridDim.y`) with their concrete values, as well as any other variables that are derived from the kernel invocation arguments.
5) If you cannot make a value concrete (e.g.: pointers), leave it as-is. Only return the transformed source code, nothing else.\n"
6) Ensure to comment the original lines that are being replaced with the new concrete values, and add the new lines below the original commented code.
7) Any lines that use blockIdx or threadIdx should have a comment below them indicating the range of values that will be used for those variables.
8) Any variables that are derived from other variables using blockIdx, threadIdx, blockDim, or gridDim should also include a comment indicating the range of values that will be used for those variables.
9) If a variable can be concretized, but is based off an expression, only fill in the variables it uses, do not evaluate the expression to a single value. 
10) Place a comment on the line below the concretized expression indicating the single value it evaluates to with an additional comment of `// Calculated value`.\n"""

def kernel_source_snippet_concretizer(state: KernelAnalysisState, config):

    # if we have some feedback messages, let's use them
    if len(state["step5_messages"]) != 0:
        msg_history = state["step5_messages"]
        concretizationState = state["snippetConcretizationState"]

        error_msg = ChatPromptTemplate.from_messages(msg_history + [
             ("assistant",
              "The concretization was rejected with the following reason(s):\n{reason}\n"
              "Please update the erroneous concretized source code with the necessary changes to make the concretization correct.\n")
        ])

        chain = error_msg | llm.with_config(configurable=config.get("configurable", {}))

        inputs = {
            "reason": concretizationState.rejectReason,
        }

        result = chain.invoke(inputs)

        snippet_kernel_src_concretized_values = result.content

        return {"snippet_kernel_src_concretized_values": snippet_kernel_src_concretized_values,
                "step5_messages": error_msg.format_messages(**inputs) + [result]}

    # this is the default path to use -- we hope the LLM agrees
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a code transformer that replaces all variable definitions, preprocessor defines, template parameters, and references in the given C/C++ CUDA kernel source code snippet with their corresponding hard-coded input argument literal values from the given kernel invocation arguments and evaluated/derived source code values."
             "Here are the rules you must follow when transforming the source code:\n"
             "{snippet_concretization_rules}\n"
             "Here is an example of the desired types of variable and explicit value concretization source code transformations:\n"
             "Example Before:\n"
             "{step5_example_before}\n\n"
             "Example After:\n"
             "{step5_example_after}\n\n"
             ),
            ("human",
                "Target Kernel Name: ```{kernel_name}```\n"
                "Kernel Invocation Arguments and Descriptions:\n{snippet_first_kernel_invocation}\n"
                "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
                "Please return the updated source code with evaluated input arguments, variables, references, template arguments, and preprocessor defines. Ensure to replace as many variables (including blockDim.x/y/z and gridDim.x/y/z variables) as possible with their literal values in the target kernel invocation call and any intermediate variables that get calculated."
                "Source code:\n{snippet_kernel_src}\n"
                )
        ])
        chain = prompt | llm.with_config(configurable=config.get("configurable", {}))

        inputs = {
            "snippet_kernel_src": state["snippet_kernel_src"],
            "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
            "kernel_name": state["kernel_name"],
            "grid_size": state["grid_size"],
            "block_size": state["block_size"], 
            "total_num_threads": state["total_num_threads"], 
            "step5_example_before": step5_example_before,
            "step5_example_after": step5_example_after,
            "snippet_concretization_rules": snippet_concretization_rules,
        }

        result = chain.invoke(inputs)

        snippet_kernel_src_concretized_values = result.content

        #print("\n\n\n")
        #print("---------- BEGIN STEP 5: Kernel Source Code Concretization ----------")
        #print(f"\n{snippet_kernel_src_concretized_values}\n")
        #print("---------- END STEP 5: Kernel Source Code Concretization ----------")
        #print("\n\n\n")

        return {"snippet_kernel_src_concretized_values": snippet_kernel_src_concretized_values,
                "step5_messages": prompt.format_messages(**inputs) + [result]}


# we're going to force this node to give us structured output (i.e: a tool call)
def snippet_concretization_checker(state: KernelAnalysisState, config):

    concretization_checker_llm = llm.with_config(configurable=config.get("configurable", {})).with_structured_output(ConcretizationChecker)

    msg_histroy = state["step5_messages"]

    # this node is used to check how well the concretization worked
    prompt = ChatPromptTemplate.from_messages([
         ("system",
          "You are a code checker that verifies the concretization of the given C/C++ CUDA source code.\n" 
          "Make sure that the returned concretized code follows the rules of the original system message.\n"
          "If the concretization follows all the rules, it is correct, and you should return the ACCEPT status tool call.\n"
          "If the concretization fails to follow at least one rule, it is incorrect, and you should return the REJECT status tool call with a brief rejectReason explaining why the concretization is incorrect or what it may be missing.\n"
          "The rejectReason should state all the possible reasons why the concretization may be incorrect."
          "Be sure to check that the produced code follows ALL the rules of the original system message. Failure to do so should result in a REJECT status.\n"
          "Below are the rules that the concretization must follow:\n"
          "{snippet_concretization_rules}\n"
          "Original system message and concretized source code response messages are provided below.\n"
          ),
    ] + msg_histroy)

    chain = prompt | concretization_checker_llm

    resultState = chain.invoke({
        "snippet_concretization_rules": snippet_concretization_rules,
    })

    return {"snippetConcretizationState": resultState}


def route_snippet_concretization_status_edge(state: KernelAnalysisState):
    """
    This function routes the edge based on the concretization status.
    If the concretization is good, it returns the next node to execute.
    If the concretization is not good, it returns the get_input_problem node to retry.
    """
    return state.get("snippetConcretizationState", {}).status












# step 6
def kernel_warp_divergence_annotator(state: KernelAnalysisState, config):

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code annotator that analyzes the given C/C++ CUDA kernel source code snippet and annotates it with warp divergence information.\n"
         "This means identifying the potential warp divergence points in the kernel code, such as conditional branches, loops, ternary, and other control flow statements that will cause threads within a warp to diverge.\n"
         "If a conditional branch is always true or always false, it is NOT considered a warp divergence point, please disregard it.\n"
         "For-loops and while-loops are always considered warp divergence points, as they can cause threads to diverge based on the loop condition.\n"
         "min and max operations are NOT considered warp divergence points, as they do not cause threads to diverge within a warp.\n"
         "Annotate the code with comments indicating where warp divergence will occur.\n"
         "The comment should appear only on conditional statements (e.g: for, if, while), and only on the line above the warp divergence point, in the format of `// WARP DIVERGENCE POINT X`, where X is the number of the warp divergence region, which starts counting at 1 and increments by 1 for each warp divergence region found.\n"
         "Do not annotate lines that are commented out.\n"
         "If an existing comment appears above a warp divergence point, add the `//WARP DIVERGENCE POINT X` annotation AFTER the existing comment.\n"
         "Code comment annotations should appear on the line immediately before the warp divergence point as in the examples below:\n"
         "If statement example:\n"
         "```// WARP DIVERGENCE POINT 1\n"
         "if (condition) {{...}}```\n\n"

         "While loop example:\n"
         "```// WARP DIVERGENCE POINT 2\n"
         "while (condition) {{...}}```\n\n"

         "For loop example:\n"
         "```// WARP DIVERGENCE POINT 3\n"
         "for (;;) {{...}}```\n\n"

         "Ternary example:\n"
         "```// WARP DIVERGENCE POINT 4\n"
         "a = b ? c : d;```\n\n"
         "Only return the annotated kernel source code, nothing else.\n"
         ),
        ("human",
            "Please return the annotated kernel source code with warp divergence indicators.\n"
            "Ensure to mark ALL allowed if statements, while loops, for loops, and ternary operators with the `// WARP DIVERGENCE POINT X` comment.\n"
            "Kernel source code:\n{snippet_kernel_src_concretized_values}\n"
            )
    ])
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
    kernel_annotated_warp_divergence = chain.invoke({
        "snippet_kernel_src_concretized_values": state["snippet_kernel_src_concretized_values"],
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 6: Kernel Warp Divergence Annotation ----------")
    print(f"\n{kernel_annotated_warp_divergence}\n")
    print("---------- END STEP 6: Kernel Warp Divergence Annotation ----------")
    print("\n\n\n")

    return {"kernel_annotated_warp_divergence": kernel_annotated_warp_divergence}









with open('./example_codes/step7_example_before.cu', 'r') as file:
    step7_example_before = file.read()
with open('./example_codes/step7_example_after.cu', 'r') as file:
    step7_example_after = file.read()

def kernel_wdp_variables_annotator(state: KernelAnalysisState, config):

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code annotator that analyzes the given C/C++ CUDA kernel source code snippet and annotates it with dependent variable range/value information for marked warp divergence regions.\n"
         "The source code contains annotations for warp divergence, indicated by `// WARP DIVERGENCE POINT` comments.\n"
         "At each warp divergence point, add a comment indicating the dependent variables used as part of the conditional branch of the warp divergence point.\n"
         "The comment should contain reasoning as to the values or range of values of the dependent variables used in the conditional statement of the warp divergence.\n"
         "Reasoning should be indicated by a comment on the line of the warp divergence point, indicated by `// WARP DIVERGENCE POINT -- VARIABLES REASONING`, followed by comments explaining explaining the values of variables the warp divergence region is dependent on.\n"
         "Code comment annotations should appear as in the example below:\n"
         "Example Before:\n{step7_example_before}\n\n"
         "Example After:\n{step7_example_after}\n\n"
         "In the above example, the added comments explain the reasoning taken to arrive at the value (or value ranges) of variables needed by the logic to enter the warp divergence region.\n"
         "Only return the annotated kernel source code, nothing else.\n"
         ),
        ("human",
            "Please return the annotated kernel source code."
            "Kernel Invocation Arguments and Descriptions:\n{snippet_first_kernel_invocation}\n\n"
            "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
            "Kernel source code:\n{kernel_annotated_warp_divergence}\n"
            )
    ])
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))

    inputs = {
        "kernel_annotated_warp_divergence": state["kernel_annotated_warp_divergence"],
        "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"],
        "total_num_threads": state["total_num_threads"],
        "step7_example_before": step7_example_before,
        "step7_example_after": step7_example_after,
    }

    result = chain.invoke(inputs)

    kernel_annotated_WDPs = result.content

    #print("\n\n\n")
    #print("---------- BEGIN STEP 7: Kernel WDP Annotation ----------")
    #print(f"\n{kernel_annotated_WDPs}\n")
    #print("---------- END STEP 7: Kernel WDP Annotation ----------")
    #print("\n\n\n")

    return {"kernel_annotated_WDPs": kernel_annotated_WDPs}










def wdp_extractor(state: KernelAnalysisState, config):

    """Extracts the warp divergence points as a list from the annotated kernel source code."""
    wdp_extractor_llm = llm.with_config(configurable=config.get("configurable", {})).with_structured_output(DivergencePointsList)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code extractor that extracts a list of warp divergence region substrings from the provided source code.\n"
         "Your job is to extract the warp divergence points from the provided source code, and return them as a list of tuples, where each tuple contains the classification of the warp divergence point and the source code of the warp divergence point.\n"
         "Below is an example of some input code:\n\n"
         "{step7_example_after}\n\n"
         "Below is an example of the captured tuples:\n"

         "WarpDivergencePoint #1:"
         "classificaiton = for\n"
        """source_code = ```// For-loop is iterating from x to y with a step of 8
// x is in [1, 1998]
// y is in [1, 998]
// for-loop entry condition: x+8*n < y, where n is an integer >= 0
for (int i = x; i < y; i += 8)```\n\n"""

         "WarpDivergencePoint #2:"
         "classification = if\n"
         """source_code = ```// IF statement is checking if x and y are within valid bounds
// x is in [0, 1999]
// y is in [0, 12409]
// entry condition: x > 0 && x < 2000-1 --> x must be in [1, 1998] --> 1998 valid x values
// entry condition: y > 0 && y < 1000-1 --> y must be in [1, 998] --> 998 valid y values
// BOTH entry conditions must be met to enter this region 
if (x > 0 && x < 2000-1 && y > 0 && y < 1000-1)```\n\n"""

         "WarpDivergencePoint #3:"
        "classification = ternary\n"
        """source_code = ```// Condition is checking if scale_factor is greater than 1000
// this condition is always true, this region will always be executed
int applyScaleFactor = (1250.0 > 1000) ? 1 : 0;```\n\n"""

         "WarpDivergencePoint #4:"
        "classification = if\n"
        """source_code = ```// Condition is always true, this region will always be executed
if (1)```\n\n"""
        "The warp divergence regions to extract are annotated with a `// WARP DIVERGENCE POINT X -- VARIABLES REASONING` comment for identification. DO NOT include the code block that the warp divergence points enclose, only the initial definition and necessary variables used in the warp divergence point entry logic.\n"
         ),
        ("human",
            "Please return a list of the warp divergence point (classification, source_code) tuples from the following source code."
            "Kernel source code:\n{kernel_annotated_WDPs}\n"
            )
    ])
    chain = prompt | wdp_extractor_llm
    wdps = chain.invoke({
        "kernel_annotated_WDPs": state["kernel_annotated_WDPs"],
        "step7_example_after": step7_example_after,
    }).warp_divergence_points

    print("\n\n\n")
    print("---------- BEGIN STEP 7a: WDP Extraction ----------")
    for wdp in wdps:
        print(f"\nclassification: [{wdp.classification}]\nsource_code:[\n{wdp.source_code}]\n\n")
    print("---------- END STEP 7a: WDP Extraction ----------")
    print("\n\n\n")

    return {"wdps_list": wdps}









# Once we have the WDPs in a list, we can query each one using o3-mini to calculate the number of times the WDP will be executed 
def wdp_num_executions_calculations(state: KernelAnalysisState, config):
    """ Calculates the number of times each warp divergence point (WDP) will be executed based on mathematical summation logic."""

    calculator_llm = llm.with_config(configurable=config.get("configurable", {})).with_structured_output(NumExecutions)

    wdps = state["wdps_list"]

    print("\n\n\n")
    print("WDPS")
    print(wdps)
    print("\n\n\n")

    print("---------- BEGIN STEP 7b: WDP Number of Operations Calculation ----------")

    calculated_executions = []

    for idx, wdp in enumerate(wdps):

        condition_type = {'if': 'if', 'else-if': 'if', 'for': 'for', 'while': 'for', 'do-while': 'for', 'ternary': 'if'}.get(wdp.classification, None)

        if condition_type is None:
            raise ValueError(f"Unsupported WDP classification: {wdp.classification}. Supported classifications are: 'if', 'else-if', 'for', 'while', 'do-while', 'ternary'.")

        if condition_type == 'if':
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are an execution analysis expert that calculates the number of times a conditional statement will be entered, given the its conditional logic and the range of values that the variables use in the conditional logic statement could take.\n"
                 "The conditional statement and it's logic is provided as a source code snippet, with the range of values of its dependent variables provided as a comment on the lines above the loop statement.\n"
                ),
                ("human",
                 "For the following loop in C++:\n"
                 "{source_code_snippet}\n\n"
                 "Explain the following:\n"
                 "1) Create a mathematical formula that calculates the number of iterations the loop will perform for any given input variables within the supplied ranges.\n"
                 "2) Create a mathematical formula that sums the total number of iterations performed for all valid input variables within the supplied ranges.\n"
                 "3) Apply and analytically evaluate the formulas (1) and (2) such that we arrive at one total sum value representing the total number of loop iterations executed by all the valid input variables within the supplied ranges.\n"
                "At each step, show your work. Return the final sum as an integer using NumExecutions num_executions. Use the value of -1 if unable to calculate an exact integer. Use a value of 0 if the loop will never execute.\n"
                "The mathematical formulas and reasoning behind the calculations should be provided in the num_executions_explanation field of the tool call.\n"
                 )
            ])
        elif condition_type == 'for':
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are an execution analysis expert that calculates the number of times a conditional loop will be executed, given the loop's conditional logic statement and the range of all possible values that the variables use in the conditional logic statement.\n"
                 "The loop and it's logic is provided as a source code snippet, and the range of values of its variables is provided as a comment on the lines above the conditional logic statement.\n"
                ),
                ("human",
                 "For the following conditional statement in C++:\n"
                 "{source_code_snippet}\n\n"
                 "Explain the following:\n"
                 "1) Create a mathematical formula that calculates the number of times the statement will be executed for any given input variables within the supplied ranges.\n"
                 "2) Create a mathematical formula that sums the total number of executions performed for all valid input variables within the supplied ranges.\n"
                 "3) Apply and analytically evaluate the formulas (1) and (2) such that we arrive at one total sum value representing the total number of executions from all the valid input variables within the supplied ranges.\n"
                "At each step, show your work. Return the final sum as an integer using NumExecutions num_executions. Use the value of -1 if unable to calculate an exact integer. Use a value of 0 if the conditional will never execute.\n"
                "The mathematical formulas and reasoning behind the calculations should be provided in the num_executions_explanation field of the tool call.\n"
                 )
            ])
        else:
            raise ValueError(f"Unsupported WDP classification for number of executions calculation: {condition_type}. Supported classifications are: 'if', 'for'.")

        chain = prompt | calculator_llm

        inputs = {
            "source_code_snippet": wdp.source_code,
        }

        result = chain.invoke(inputs)

        num_executions = result.num_executions

        print("\n")
        print(f"\t\t [{idx+1}] ({condition_type}) Number of Executions Calculation: [{num_executions}]") 
        print("\n")

        calculated_executions.append(result)

    print("---------- END STEP 7b: WDP Number of Operations Calculation ----------")

    print("wdps_num_executions:", calculated_executions)

    return {"wdps_num_executions": calculated_executions}









#with open('./example_codes/step8_example_before.cu', 'r') as file:
#    step8_example_before = file.read()
#with open('./example_codes/step8_example_after.cu', 'r') as file:
#    step8_example_after = file.read()
with open('./example_codes/step8_examples.cu', 'r') as file:
    step8_examples = file.read()

kernel_num_ops_rules = """
1) Identify the number of SP-FLOP and DP-FLOP operations performed. 
2) If a line is performing floating point operations, add a comment on the line above it indicating the number of operations performed.
3) This comment should be followed by an explanation as to the number of FLOP operations for that line.
4) If a fused-multiply-add (FMA) operation is encountered, count it as 2 operations (1 for the multiply and 1 for the add).
5) If a loop with logic that performs floating point operations is encountered, annotate the number of operations performed by the loop continuation logic.
6) DO NOT comment or annotate lines that are not performing floating point operations.
7) Only consider arithmetic operations (ADD, SUB, MUL, DIV, FMA) involving floating point numbers.
8) DO NOT consider the number of threads during execution, instead assume a single thread of execution for the purpose of counting operations.\n\n"""

def kernel_num_ops_annotator(state: KernelAnalysisState, config):

    # if we have some feedback messages, let's use them
    if len(state["step8_messages"]) != 0:
        msg_history = state["step8_messages"]
        annotationState = state["numOpsAnnotationState"]

        error_msg = ChatPromptTemplate.from_messages(msg_history + [
             ("assistant",
              "The concretization was rejected with the following reason(s):\n{reason}\n"
              "Please update the erroneous concretized source code with the necessary changes to make the concretization correct.\n")
        ])

        chain = error_msg | llm.with_config(configurable=config.get("configurable", {}))

        inputs = {
            "reason": annotationState.rejectReason,
        }

        result = chain.invoke(inputs)

        kernel_annotated_num_ops = result.content

        return {"kernel_annotated_num_ops": kernel_annotated_num_ops,
                "step8_messages": error_msg.format_messages(**inputs) + [result]}

    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a code annotator that analyzes the given C/C++ CUDA kernel source code snippet and annotates it with the number of single-precision (SP-FLOP) and double-precision (DP-FLOP) floating point operations performed at each part of the kernel code. "
             "For each line of the source code that performs floating point operations, follow these rules:\n"
             "{kernel_num_ops_rules}\n"
             "Only return the annotated kernel source code, nothing else.\n"
             "Code annotations and comments should appear in the format of the example below:\n"
             "Examples:\n{step8_examples}\n\n"
             ),
            ("human",
                "Kernel Invocation Arguments and Descriptions:\n{snippet_first_kernel_invocation}\n\n"
                "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
                "Kernel source code:\n{snippet_kernel_src_concretized_values}\n"
                )
        ])
        chain = prompt | llm.with_config(configurable=config.get("configurable", {}))

        inputs = {
            "snippet_kernel_src_concretized_values": state["snippet_kernel_src_concretized_values"],
            "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
            "grid_size": state["grid_size"],
            "block_size": state["block_size"], 
            "total_num_threads": state["total_num_threads"],
            "step8_examples": step8_examples,
            "kernel_num_ops_rules": kernel_num_ops_rules,
        }

        result = chain.invoke(inputs)

        kernel_annotated_num_ops = result.content

        #print("\n\n\n")
        #print("---------- BEGIN STEP 8: Kernel Number of Operations Annotation ----------")
        #print(f"\n{kernel_annotated_num_ops}\n")
        #print("---------- END STEP 8: Kernel Number of Operations Annotation ----------")
        #print("\n\n\n")

        return {"kernel_annotated_num_ops": kernel_annotated_num_ops,
                "step8_messages": prompt.format_messages(**inputs) + [result]}


def num_ops_checker(state: KernelAnalysisState, config):

    num_ops_checker_llm = llm.with_config(configurable=config.get("configurable", {})).with_structured_output(NumOpsState)

    msg_histroy = state["step8_messages"]

    # this node is used to check how well the concretization worked
    prompt = ChatPromptTemplate.from_messages([
         ("system",
          "You are a code checker that verifies the SP-FLOP and DP-FLOP floating point operation count annotations for a given C/C++ CUDA source code.\n" 
          "Make sure that the returned code follows the rules of the original system message.\n"
          "If the annotated code follows all the rules, it is correct, and you should return the ACCEPT status tool call.\n"
          "If the annotated fails to follow at least one rule, it is incorrect, and you should return the REJECT status tool call with a brief rejectReason explaining why the annotations are incorrect or what it may be missing.\n"
          "The rejectReason should explain why the 'REJECT' status was given, and state all the possible reasons why the floating point operation counts may be incorrect."
          "Be sure to check that the produced code follows ALL the rules of the original system message. Failure to do so should result in a REJECT status.\n"
          "The acceptReason should explain why the 'ACCEPT' status was given, and why the annotations are correct and how they follow the rules of the original system message.\n"
          "Below are the rules that the annotations must follow:\n"
          "{kernel_num_ops_rules}\n"
          "Original system message and source code response messages are provided below.\n"
          ),
    ] + msg_histroy)

    chain = prompt | num_ops_checker_llm 

    resultState = chain.invoke({
        "kernel_num_ops_rules": kernel_num_ops_rules,
    })

    return {"numOpsAnnotationState": resultState}


def route_num_ops_annotation_status_edge(state: KernelAnalysisState) -> Literal["ACCEPT", "REJECT"]:
    return state.get("numOpsAnnotationState", {}).status











def kernel_ops_summarizer(state: KernelAnalysisState, config):

    wdps_list = state["wdps_list"]

    num_executions = state["wdps_num_executions"]

    wdps = zip(wdps_list, num_executions)

    wdps_string = ""

    for wdp, num_executions in wdps:
        if num_executions.num_executions <= 0:
            wdps_string += f"\n{wdp.source_code.strip()}\n Could not calculate the exact number of executions for this warp divergence point.\n\n"
        else:
            wdps_string += f"\n{wdp.source_code.strip()}\nNumber of ({wdp.classification}) executions: {num_executions.num_executions}\n\n"

    print("WDPS STRING", wdps_string)

    flop_counts_llm = llm.with_config(configurable=config.get("configurable", {})).with_structured_output(FLOPCounts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a CUDA kernel FLOP count calculator that sums the total number of single-precision (SP-FLOP) and double-precision (DP-FLOP) floating point operations performed by the given C/C++ CUDA kernel source code snippet.\n"
         "You will be given two annotated source code snippets: one with warp divergence points annotated with the number of executions they will perform across all threads, and another with the number of FLOP operations performed at each line of the kernel code.\n"
         "Use the annotated codes to sum up the total number of operations performed by the kernel, accounting for the number of threads that will enter each warp divergence point.\n"
         "The output should briefly explain how it arrived at the total number of SP-FLOP and DP-FLOP operations performed by the kernel.\n"
         "At the end of the summary, return the final summed counts using a tool call with sp_flop_count and dp_flop_count representing the total sums of the single and double precision floating point operations, respectively."
         "The logic/explanations for how the total number of operations were calculated should be included as sp_flop_explanation and dp_flop_explanation in the tool call.\n"
         ),
        ("human",
            "Kernel Invocation Arguments and Descriptions:\n{snippet_first_kernel_invocation}\n\n"
            "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
            "Kernel source code with SP-FLOP, and DP-FLOP annotations:\n{kernel_annotated_num_ops}\n"
            "Kernel source code warp divergence snippets and their associated number of execution counts:\n{wdps_string}\n\n"
            )
    ])
    chain = prompt | flop_counts_llm

    inputs = {
        "kernel_annotated_num_ops": state["kernel_annotated_num_ops"],
        "wdps_string": wdps_string,
        "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"], 
        "total_num_threads": state["total_num_threads"],
    }

    result = chain.invoke(inputs)

    summed_kernel_ops = result

    print("\n\n\n")
    print("---------- BEGIN STEP 9: Kernel Operations Summary ----------")
    print(f"\n{summed_kernel_ops}\n")
    print("---------- END STEP 9: Kernel Operations Summary ----------")
    print("\n\n\n")

    empirical_sp_flop_count = state["empirical_sp_flop_count"]
    empirical_dp_flop_count = state["empirical_dp_flop_count"]

    sp_flop_diff = summed_kernel_ops.sp_flop_count - empirical_sp_flop_count
    dp_flop_diff = summed_kernel_ops.dp_flop_count - empirical_dp_flop_count

    sp_flop_perc_diff = ((sp_flop_diff * 100) / empirical_sp_flop_count) if empirical_sp_flop_count != 0 else 0
    dp_flop_perc_diff = ((dp_flop_diff * 100) / empirical_dp_flop_count) if empirical_dp_flop_count != 0 else 0

    return {"summed_kernel_ops": summed_kernel_ops,
            "sp_flop_diff": sp_flop_diff,
            "dp_flop_diff": dp_flop_diff,
            "sp_flop_perc_diff": sp_flop_perc_diff,
            "dp_flop_perc_diff": dp_flop_perc_diff,
            }
