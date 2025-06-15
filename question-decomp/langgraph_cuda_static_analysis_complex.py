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
from langchain_core.runnables.graph import MermaidDrawMethod


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
  model_name="openai/gpt-4.1-mini"
  #model_name="openai/gpt-4o-mini", # cheap model for testing
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
    total_num_threads: str

    # these will be filled in by the nodes
    src_concretized_input_args: str
    src_single_kernel_execution: str
    snippet_first_kernel_invocation: str
    snippet_kernel_src: str
    snippet_kernel_src_concretized_values: str
    kernel_annotated_warp_divergence: str
    kernel_annotated_num_threads: str
    kernel_annotated_num_ops: str
    summed_kernel_ops: str







with open('./example_codes/step1_example_before.cu', 'r') as file:
        step1_example_before = file.read()
with open('./example_codes/step1_example_after.cu', 'r') as file:
        step1_example_after = file.read()

def src_input_args_concretizer(state: KernelAnalysisState, llm: ChatOpenAI):

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code transformer that replaces all variable definitions, preprocessor defines, template parameters, and references in the given C/C++ CUDA source code with their corresponding hard-coded input argument literal values from the given execution arguments and evaluated/derived source code values.\n"
         "When a value is being concretized/replaced, make sure to comment out the original line that is being replaced with the new concrete value, and add the new line below the original commented code.\n"
         "Show the steps taken in calculating any values as commented lines.\n"
         "Only replace the variables in arightmetic expressions, and include a comment below any variables that are calculated to single values, indicated by a `// Calculated value(s)` comment.\n"
         "If a value is derived from other value(s), show the steps to arrive at the hard-coded value. For CUDA kernel invocations, make sure all possible kernel input arguments are made explicit using the concrete values.\n"
         "If you cannot make a value concrete (e.g.: pointers), leave it as-is. Only return the transformed source code, nothing else."
         "Do not include any additional comments or explanations in the output.\n"
         "If the `auto` keyword is used, replace it with the corresponding concrete type.\n"
         "If a ternary operator is encountered, convert it to a regular if statement and mark the conversion with the comment of `// CONVERTED TERNARY TO IF STATEMENT`.\n"
         "Below is an example of the desired types of variable and explicit value concretization source code transformations:\n"
         "{step1_example_before}\n\n"
         "{step1_example_after}\n\n"),
        ("human", 
         "Target Kernel Name: {kernel_name}\n"
         "Execution Arguments: {exec_args}\n"
         "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
         "Please return the updated source code with evaluated input arguments, variables, references, template arguments, and preprocessor defines. Ensure to calculate as many variables as possible with their literal values in the target kernel invocation call and any intermediate variables that get calculated."
         "Source code:\n```{source_code}```\n\n"
         )
    ])
    chain = prompt | llm
    updated_source = chain.invoke({
        "source_code": state["source_code"],
        "kernel_name": state["kernel_name"],
        "exec_args": state["exec_args"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"], 
        "total_num_threads": state["total_num_threads"], 
        "step1_example_before": step1_example_before,
        "step1_example_after": step1_example_after,
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 1: Source Code Concretization ----------")
    print(f"\n{updated_source}\n")
    print("---------- END STEP 1: Source Code Concretization ----------")
    print("\n\n\n")

    return {"src_concretized_input_args": updated_source}


# Node wrappers for LangGraph
def make_src_input_args_concretizer_node(llm):
    def node(state):
        return src_input_args_concretizer(state, llm)
    return node







with open('./example_codes/step2_example_before.cu', 'r') as file:
    step2_example_before = file.read()

with open('./example_codes/step2_example_after.cu', 'r') as file:
    step2_example_after = file.read()


def src_single_kernel_execution_modifier(state: KernelAnalysisState, llm: ChatOpenAI): 
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code transformer that modifies the given C/C++ CUDA source code to ensure that only a single kernel invocation of the target kernel name is executed. "
         "This means removing any loops or multiple invocations of the kernel, to leave only the first invocation of the target kernel in the code. This also means ensuring that the kernel is invoked with the correct arguments, grid, and block sizes."
         "The modifications should be done by commenting out parts of the original code to be changed, and adding the changes on a new line below the original commented code."
         "Only return the modified source code, nothing else.\nAn example is provided below:\n"
         "Example Before:\n"
         "Target Kernel Name: example_kernel\n"
         "{step2_example_before}\n\n"
         "Example After:\n"
         "{step2_example_after}\n\n"),
        ("human", 
         "Target Kernel Name: {kernel_name}\n"
         #"Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
         "Please return the updated source code with only a single kernel invocation."
         "Source code:\n{updated_source}\n"
         )
    ])
    chain = prompt | llm
    single_kernel_source = chain.invoke({
        "updated_source": state["src_concretized_input_args"],
        "kernel_name": state["kernel_name"],
        #"grid_size": state["grid_size"],
        #"block_size": state["block_size"], 
        #"total_num_threads": state["total_num_threads"], 
        "step2_example_before": step2_example_before,
        "step2_example_after": step2_example_after,
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 2: Single Kernel Execution Modification ----------")
    print(f"\n{single_kernel_source}\n")
    print("---------- END STEP 2: Single Kernel Execution Modification ----------")
    print("\n\n\n")

    return {"src_single_kernel_execution": single_kernel_source}

# Node wrappers for LangGraph
def make_src_single_kernel_execution_modifier_node(llm):
    def node(state):
        return src_single_kernel_execution_modifier(state, llm)
    return node









def first_kernel_invocation_snippet_extractor(state: KernelAnalysisState, llm: ChatOpenAI):
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
    chain = prompt | llm
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

# Node wrappers for LangGraph
def make_first_kernel_invocation_snippet_extractor_node(llm):
    def node(state):
        return first_kernel_invocation_snippet_extractor(state, llm)
    return node









def kernel_source_snippet_extractor(state: KernelAnalysisState, llm: ChatOpenAI):
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
    chain = prompt | llm
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

# Node wrappers for LangGraph
def make_kernel_source_snippet_extractor_node(llm):
    def node(state):
        return kernel_source_snippet_extractor(state, llm)
    return node








with open('./example_codes/step5_example_before.cu', 'r') as file:
    step5_example_before = file.read()

with open('./example_codes/step5_example_after.cu', 'r') as file:
    step5_example_after = file.read()

def kernel_source_snippet_concretizer(state: KernelAnalysisState, llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code transformer that replaces all variable definitions, preprocessor defines, template parameters, and references in the given C/C++ CUDA kernel source code snippet with their corresponding hard-coded input argument literal values from the given kernel invocation arguments and evaluated/derived source code values."
         "If a value is derived from other value(s), also replace it with the hard-coded value. Make sure all possible variables and arguments are made explicit using the provided hard-coded values and their descriptions. "
         "If the `auto` keyword is used, replace it with the correct concrete type."
         "Be sure to replace any blockDim and gridDim variables (e.g: `blockDim.x` or `gridDim.y`) with their concrete values, as well as any other variables that are derived from the kernel invocation arguments."
         "If you cannot make a value concrete (e.g.: pointers), leave it as-is. Only return the transformed source code, nothing else.\n"
         "Ensure to comment the original lines that are being replaced with the new concrete values, and add the new lines below the original commented code.\n"
         "Any lines that use blockIdx or threadIdx should have a comment below them indicating the range of values that will be used for those variables.\n"
         "If a variable can be concretized, but is based off an expression, only fill in the variables it uses, do not evaluate the expression to a single value. Place a comment on the line below the concretized expression indicating the single value it evaluates to with an additional comment of `// Calculated value`.\n"
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
            "Please return the updated source code with evaluated input arguments, variables, references, template arguments, and preprocessor defines. Ensure to replace as many variables (including blockDim and gridDim variables) as possible with their literal values in the target kernel invocation call and any intermediate variables that get calculated."
            "Source code:\n{snippet_kernel_src}\n"
            )
    ])
    chain = prompt | llm
    snippet_kernel_src_concretized_values = chain.invoke({
        "snippet_kernel_src": state["snippet_kernel_src"],
        "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
        "kernel_name": state["kernel_name"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"], 
        "total_num_threads": state["total_num_threads"], 
        "step5_example_before": step5_example_before,
        "step5_example_after": step5_example_after,
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 5: Kernel Source Code Concretization ----------")
    print(f"\n{snippet_kernel_src_concretized_values}\n")
    print("---------- END STEP 5: Kernel Source Code Concretization ----------")
    print("\n\n\n")

    return {"snippet_kernel_src_concretized_values": snippet_kernel_src_concretized_values}

# Node wrappers for LangGraph
def make_kernel_source_snippet_concretizer_node(llm):
    def node(state):
        return kernel_source_snippet_concretizer(state, llm)
    return node





def kernel_warp_divergence_annotator(state: KernelAnalysisState, llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code annotator that analyzes the given C/C++ CUDA kernel source code snippet and annotates it with warp divergence information.\n"
         "This means identifying the potential warp divergence points in the kernel code, such as conditional branches, loops, ternary, and other control flow statements that will cause threads within a warp to diverge.\n"
         "If a conditional branch is always true or always false, it is still considered a warp divergence point.\n"
         "min and max operations are not considered warp divergence points, as they do not cause threads to diverge within a warp.\n"
         "Annotate the code with comments indicating where warp divergence will occur.\n"
         "Code comment annotations should appear on the line before as in the examples below:\n"
         "If statement example:\n"
         "// WARP DIVERGENCE POINT\n"
         "if (condition) {{...}}\n\n"

         "While loop example:\n"
         "// WARP DIVERGENCE POINT\n"
         "while (condition) {{...}}\n\n"

         "For loop example:\n"
         "// WARP DIVERGENCE POINT\n"
         "for (;;) {{...}}\n\n"

         "Ternary example:\n"
         "// WARP DIVERGENCE POINT\n"
         "a = b ? c : d;\n"
         "Only return the annotated kernel source code, nothing else.\n"
         ),
        ("human",
            "Please return the annotated kernel source code with warp divergence indicators.\n"
            "Ensure to mark ALL if statements, while loops, for loops, and ternary operators with the `// WARP DIVERGENCE POINT` comment.\n"
            "Source code:\n{snippet_kernel_src_concretized_values}\n"
            )
    ])
    chain = prompt | llm
    kernel_annotated_warp_divergence = chain.invoke({
        "snippet_kernel_src_concretized_values": state["snippet_kernel_src_concretized_values"],
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 6: Kernel Warp Divergence Annotation ----------")
    print(f"\n{kernel_annotated_warp_divergence}\n")
    print("---------- END STEP 6: Kernel Warp Divergence Annotation ----------")
    print("\n\n\n")

    return {"kernel_annotated_warp_divergence": kernel_annotated_warp_divergence}

# Node wrappers for LangGraph
def make_kernel_warp_divergence_annotator_node(llm):
    def node(state):
        return kernel_warp_divergence_annotator(state, llm)
    return node








with open('./example_codes/step7_example_before.cu', 'r') as file:
    step7_example_before = file.read()
with open('./example_codes/step7_example_after.cu', 'r') as file:
    step7_example_after = file.read()

def kernel_num_threads_annotator(state: KernelAnalysisState, llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code annotator that analyzes the given C/C++ CUDA kernel source code snippet and annotates it with the number of threads that will execute at each part of the kernel code.\n"
         "The source code contains annotations for warp divergence, indicated by `// WARP DIVERGENCE POINT` comments.\n"
         "At each warp divergence point, add a comment identifying the TOTAL number of threads that will enter or not enter the warp divergence point due to the conditional branch encountered.\n"
         "The comment should contain reasoning as to the total number of threads entering the warp divergence point.\n"
         "Reasoning should be indicated by a comment on the line of the warp divergence point, indicated by `// WARP DIVERGENCE POINT -- TOTAL THREADS ENTERING REASONING`, followed by comments explaining the reasoning for the number of threads that will enter the warp divergence point.\n"
         "At the end of the reasoning, add a comment indicating the total number of threads that will enter the warp divergence point, indicated by `// WARP DIVERGENCE POINT -- TOTAL NUM THREADS ENTERING REGION: XXX`.\n"
         "Where XXX is the total number of threads that will enter the warp divergence region.\n"
         "Code comment annotations should appear as in the example below:\n"
         "Example Before:\n{step7_example_before}\n\n"
         "Example After:\n{step7_example_after}\n\n"
         "In the above example, the added comments explain the reasoning taken to arrive at the number of threads that enter a warp divergence region.\n"
         "Only return the annotated kernel source code, nothing else.\n"
         ),
        ("human",
            "Please return the annotated kernel source code with thread count indicators."
            "Kernel Invocation Arguments and Descriptions:\n{snippet_first_kernel_invocation}\n\n"
            "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
            "Source code:\n{kernel_annotated_warp_divergence}\n"
            )
    ])
    chain = prompt | llm
    kernel_annotated_num_threads = chain.invoke({
        "kernel_annotated_warp_divergence": state["kernel_annotated_warp_divergence"],
        "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"],
        "total_num_threads": state["total_num_threads"],
        "step7_example_before": step7_example_before,
        "step7_example_after": step7_example_after,
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 7: Kernel Number of Threads Annotation ----------")
    print(f"\n{kernel_annotated_num_threads}\n")
    print("---------- END STEP 7: Kernel Number of Threads Annotation ----------")
    print("\n\n\n")

    return {"kernel_annotated_num_threads": kernel_annotated_num_threads}

# Node wrappers for LangGraph
def make_kernel_num_threads_annotator_node(llm):
    def node(state):
        return kernel_num_threads_annotator(state, llm)
    return node






#with open('./example_codes/step8_example_before.cu', 'r') as file:
#    step8_example_before = file.read()
#with open('./example_codes/step8_example_after.cu', 'r') as file:
#    step8_example_after = file.read()
with open('./example_codes/step8_examples.cu', 'r') as file:
    step8_examples = file.read()

def kernel_num_ops_annotator(state: KernelAnalysisState, llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         #"You are a code annotator that analyzes the given C/C++ CUDA kernel source code snippet and annotates it with the number of integer (INTOP), single-precision (SP-FLOP), and double-precision (DP-FLOP) floating point operations performed at each part of the kernel code. "
         "You are a code annotator that analyzes the given C/C++ CUDA kernel source code snippet and annotates it with the number of single-precision (SP-FLOP) and double-precision (DP-FLOP) floating point operations performed at each part of the kernel code. "
         "For each line of the source code, identify the number of SP-FLOP and DP-FLOP operations performed. If a line is performing floating point operations, add a comment on the line above it indicating the number of operations performed. This comment should be followed by an explanation as to the number of FLOP operations for that line. \n"
         "If a fused-multiply-add (FMA) operation is encountered, count it as 2 operations (1 for the multiply and 1 for the add).\n"
         "If a loop with logic that performs floating point operations is encountered, annotate the number of operations performed by the loop continuation logic.\n"
         "DO NOT comment or annotate lines that are not performing floating point operations.\n"
         "Only consider arithmetic operations (ADD, SUB, MUL, DIV, FMA).\n"
         "DO NOT consider the number of threads during execution, instead assume a single thread of execution for the purpose of counting operations.\n"
         "Only return the annotated kernel source code, nothing else.\n"
         "Code annotations and comments should appear in the format of the example below:\n"
         "Examples:\n{step8_examples}\n\n"
         #"Example Before:\n{step8_example_before}\n\n"
         #"Example After:\n{step8_example_after}\n\n"
         ),
        ("human",
            "Kernel Invocation Arguments and Descriptions:\n{snippet_first_kernel_invocation}\n\n"
            "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
            "Source code:\n{snippet_kernel_src_concretized_values}\n"
            )
    ])
    chain = prompt | llm
    kernel_annotated_num_ops = chain.invoke({
        "snippet_kernel_src_concretized_values": state["snippet_kernel_src_concretized_values"],
        "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"], 
        "total_num_threads": state["total_num_threads"],
        #"step8_example_before": step8_example_before,
        #"step8_example_after": step8_example_after,
        "step8_examples": step8_examples,
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 8: Kernel Number of Operations Annotation ----------")
    print(f"\n{kernel_annotated_num_ops}\n")
    print("---------- END STEP 8: Kernel Number of Operations Annotation ----------")
    print("\n\n\n")

    return {"kernel_annotated_num_ops": kernel_annotated_num_ops}

# Node wrappers for LangGraph
def make_kernel_num_ops_annotator_node(llm):
    def node(state):
        return kernel_num_ops_annotator(state, llm)
    return node




def kernel_ops_summarizer(state: KernelAnalysisState, llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         #"You are a code summarizer that summarizes the number of integer (INTOP), single-precision (SP-FLOP), and double-precision (DP-FLOP) floating point operations performed by the given C/C++ CUDA kernel source code snippet.\n"
         "You are a code summarizer that summarizes the number of single-precision (SP-FLOP) and double-precision (DP-FLOP) floating point operations performed by the given C/C++ CUDA kernel source code snippet.\n"
         "You will be given two annotated source code snippets: one with warp divergence points and the number of threads that will execute at each part of the kernel code, and another with the number of operations performed at each line of the kernel code.\n"
         "Use the annotated codes to sum up the total number of operations performed by the kernel, accounting for the number of threads that will enter each warp divergence point.\n"
         "Only return the summary in the format as in the example below:\n"
         "SP-FLOP: YYY\nDP-FLOP: ZZZ\n"
         ),
        ("human",
            "Kernel Invocation Arguments and Descriptions:\n{snippet_first_kernel_invocation}\n\n"
            "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
            "Source Code with INTOP, SP-FLOP, and DP-FLOP annotations:\n{kernel_annotated_num_ops}\n"
            "Source Code with warp divergence and thread count annotations:\n{kernel_annotated_num_threads}\n\n"
            )
    ])
    chain = prompt | llm
    summed_kernel_ops = chain.invoke({
        "kernel_annotated_num_ops": state["kernel_annotated_num_ops"],
        "kernel_annotated_num_threads": state["kernel_annotated_num_threads"],
        "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"], 
        "total_num_threads": state["total_num_threads"],
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 9: Kernel Operations Summary ----------")
    print(f"\n{summed_kernel_ops}\n")
    print("---------- END STEP 9: Kernel Operations Summary ----------")
    print("\n\n\n")

    return {"summed_kernel_ops": summed_kernel_ops}

# Node wrappers for LangGraph
def make_kernel_ops_summarizer_node(llm):
    def node(state):
        return kernel_ops_summarizer(state, llm)
    return node



# Build the graph
def build_cuda_kernel_ops_graph(llm, show_mermaid_png: bool = False):
    workflow = StateGraph(KernelAnalysisState)

    workflow.add_node("src_input_args_concretizer_1", make_src_input_args_concretizer_node(llm))
    workflow.add_node("src_single_kernel_execution_modifier_2", make_src_single_kernel_execution_modifier_node(llm))
    workflow.add_node("first_kernel_invocation_snippet_extractor_3", make_first_kernel_invocation_snippet_extractor_node(llm))
    workflow.add_node("kernel_source_snippet_extractor_4", make_kernel_source_snippet_extractor_node(llm))
    workflow.add_node("kernel_source_snippet_concretizer_5", make_kernel_source_snippet_concretizer_node(llm))
    workflow.add_node("kernel_warp_divergence_annotator_6", make_kernel_warp_divergence_annotator_node(llm))
    workflow.add_node("kernel_num_threads_annotator_7", make_kernel_num_threads_annotator_node(llm))
    workflow.add_node("kernel_num_ops_annotator_8", make_kernel_num_ops_annotator_node(llm))
    workflow.add_node("kernel_ops_summarizer_9", make_kernel_ops_summarizer_node(llm))


    # Graph edges
    workflow.add_edge("src_input_args_concretizer_1", "src_single_kernel_execution_modifier_2")

    workflow.add_edge("src_single_kernel_execution_modifier_2", "first_kernel_invocation_snippet_extractor_3")

    workflow.add_edge("first_kernel_invocation_snippet_extractor_3", "kernel_source_snippet_extractor_4")

    workflow.add_edge(["kernel_source_snippet_extractor_4","first_kernel_invocation_snippet_extractor_3"],  "kernel_source_snippet_concretizer_5")

    workflow.add_edge("kernel_source_snippet_concretizer_5", "kernel_warp_divergence_annotator_6")

    workflow.add_edge(["kernel_warp_divergence_annotator_6","first_kernel_invocation_snippet_extractor_3"],  "kernel_num_threads_annotator_7")

    workflow.add_edge(["first_kernel_invocation_snippet_extractor_3","kernel_source_snippet_concretizer_5"], "kernel_num_ops_annotator_8")

    workflow.add_edge(["kernel_num_threads_annotator_7", "kernel_num_ops_annotator_8", "first_kernel_invocation_snippet_extractor_3"], "kernel_ops_summarizer_9")

    workflow.add_edge("kernel_ops_summarizer_9", END)

    # Set entrypoint
    workflow.set_entry_point("src_input_args_concretizer_1")
    compiled = workflow.compile()

    # Draw and save the graph as a PNG image if requested
    if show_mermaid_png:
        #display(Image(compiled.get_graph().draw_png()))
        display(Image(compiled.get_graph().draw_mermaid_png()))
        #display(Image(compiled.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)))
        #compiled.get_graph().print_ascii()

    return compiled

# -- Example usage for initializing the state from user data --


# Usage Example:
graph = build_cuda_kernel_ops_graph(llm, show_mermaid_png=True)
#result = graph.invoke(query_data)

#print(result['final_op_counts'])
