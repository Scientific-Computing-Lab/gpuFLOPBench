from utils.state import KernelAnalysisState, WarpDivergencePoint

from typing_extensions import TypedDict, List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

#from langchain.chat_models import init_chat_model
from langchain_core.runnables import ConfigurableField

from .dataset import df

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

# Calculate the total number of threads from the gridSz and the blockSz
# grid size is a string of format "(x, y, z)"
# block size is a string of format "(x, y, z)"
def calc_total_threads(gridSz:str, blockSz:str):
    gridSz = eval(gridSz)
    blockSz = eval(blockSz)
    total_threads = gridSz[0] * gridSz[1] * gridSz[2] * blockSz[0] * blockSz[1] * blockSz[2]
    return str(total_threads)



def get_input_problem(state: KernelAnalysisState, config):

    target_name = config.get("configurable", {}).get("input_problem", "resize-cuda")

    row = df[df['targetName'] == target_name].iloc[0]

    assert row is not None, f"Target problem '{target_name}' not found in the dataset."

    return {'source_code' : row['kernelCode'],
            'kernel_name' : row['Kernel Name'],
            'exec_args' : row['exeArgs'],
            'grid_size' : row['Grid Size'],
            'block_size' : row['Block Size'],
            'total_num_threads' : calc_total_threads(row['Grid Size'], row['Block Size']),
            }







with open('./example_codes/step1_example_before.cu', 'r') as file:
        step1_example_before = file.read()
with open('./example_codes/step1_example_after.cu', 'r') as file:
        step1_example_after = file.read()

def src_input_args_concretizer(state: KernelAnalysisState, config):

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
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
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








with open('./example_codes/step2_example_before.cu', 'r') as file:
    step2_example_before = file.read()

with open('./example_codes/step2_example_after.cu', 'r') as file:
    step2_example_after = file.read()


def src_single_kernel_execution_modifier(state: KernelAnalysisState, config): 

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
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
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

def kernel_source_snippet_concretizer(state: KernelAnalysisState, config):

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
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
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






def kernel_warp_divergence_annotator(state: KernelAnalysisState, config):

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a code annotator that analyzes the given C/C++ CUDA kernel source code snippet and annotates it with warp divergence information.\n"
         "This means identifying the potential warp divergence points in the kernel code, such as conditional branches, loops, ternary, and other control flow statements that will cause threads within a warp to diverge.\n"
         "If a conditional branch is always true or always false, it is still considered a warp divergence point.\n"
         "min and max operations are not considered warp divergence points, as they do not cause threads to diverge within a warp.\n"
         "Annotate the code with comments indicating where warp divergence will occur.\n"
         "The comment should appear only on conditional statements, and only on the line above the warp divergence point, in the format of `// WARP DIVERGENCE POINT`.\n"
         "Do not annotate lines that are commented out.\n"
         "If an existing comment appears above a warp divergence point, add the `//WARP DIVERGENCE POINT` annotation AFTER the existing comment.\n"
         "Code comment annotations should appear on the line immediately before the warp divergence point as in the examples below:\n"
         "If statement example:\n"
         "```// WARP DIVERGENCE POINT\n"
         "if (condition) {{...}}```\n\n"

         "While loop example:\n"
         "```// WARP DIVERGENCE POINT\n"
         "while (condition) {{...}}```\n\n"

         "For loop example:\n"
         "```// WARP DIVERGENCE POINT\n"
         "for (;;) {{...}}```\n\n"

         "Ternary example:\n"
         "```// WARP DIVERGENCE POINT\n"
         "a = b ? c : d;```\n\n"
         "Only return the annotated kernel source code, nothing else.\n"
         ),
        ("human",
            "Please return the annotated kernel source code with warp divergence indicators.\n"
            "Ensure to mark ALL if statements, while loops, for loops, and ternary operators with the `// WARP DIVERGENCE POINT` comment.\n"
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
    kernel_annotated_WDPs = chain.invoke({
        "kernel_annotated_warp_divergence": state["kernel_annotated_warp_divergence"],
        "snippet_first_kernel_invocation": state["snippet_first_kernel_invocation"],
        "grid_size": state["grid_size"],
        "block_size": state["block_size"],
        "total_num_threads": state["total_num_threads"],
        "step7_example_before": step7_example_before,
        "step7_example_after": step7_example_after,
    }).content

    print("\n\n\n")
    print("---------- BEGIN STEP 7: Kernel WDP Annotation ----------")
    print(f"\n{kernel_annotated_WDPs}\n")
    print("---------- END STEP 7: Kernel WDP Annotation ----------")
    print("\n\n\n")

    return {"kernel_annotated_WDPs": kernel_annotated_WDPs}







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

def wdp_extractor(state: KernelAnalysisState, config):

    """Extracts the warp divergence points as a list from the annotated kernel source code."""
    wdp_extractor_llm = llm.with_structured_output(DivergencePointsList)

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
        "The warp divergence regions to extract are annotated with a `// WARP DIVERGENCE POINT -- VARIABLES REASONING` comment for identification. DO NOT include the code block that the warp divergence points enclose, only the initial definition and necessary variables used in the warp divergence point entry logic.\n"
         ),
        ("human",
            "Please return a list of the warp divergence point (classification, source_code) tuples from the following source code."
            "Kernel source code:\n{kernel_annotated_WDPs}\n"
            )
    ])
    chain = prompt | wdp_extractor_llm.with_config(configurable=config.get("configurable", {}))
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








# We create a custom structured output so that models return the number of iterations executed for each warp divergence point
class NumExecutions(BaseModel):
    num_executions: int = Field(..., description="Calculated number of times the source code will be executed based on the mathematical summation logic provided in the prompt. This is a single integer value representing the total number of times the given code snippet will be executed for the provided conditional values. -1 indicates that we are unable to calculate an exact integer number of executions.")

# Once we have the WDPs in a list, we can query each one using o3-mini to calculate the number of times the WDP will be executed 
def wdp_num_executions_calculations(state: KernelAnalysisState, config):
    """ Calculates the number of times each warp divergence point (WDP) will be executed based on mathematical summation logic."""

    calculator_llm = llm.with_structured_output(NumExecutions)

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
                "At each step, show your work. Return the final sum as an integer using NumExecutions num_executions. Use the value of -1 if unable to calculate an exact integer. Use a value of -999 if the loop will always execute. Use a value of 0 if the loop will never execute.\n"
                 )
            ])
            pass
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
                 "3) Apply and analytically evaluate the formulas (1) and (2) such that we arrive at one total sum value representing the total number of loop iterations executed by all the valid input variables within the supplied ranges.\n"
                "At each step, show your work. Return the final sum as an integer using NumExecutions num_executions. Use the value of -1 if unable to calculate an exact integer. Use a value of -999 if the conditional will always execute. Use a value of 0 if the conditional will never execute.\n"
                 )
            ])

        chain = prompt | calculator_llm.with_config(configurable=config.get("configurable", {}))
        num_executions = chain.invoke({
            "source_code_snippet": wdp.source_code,
        }).num_executions

        print("\n")
        print(f"\t\t [{idx+1}] ({condition_type}) Number of Executions Calculation {num_executions}") 
        print("\n")

        calculated_executions.append(num_executions)

    print("---------- END STEP 7b: WDP Number of Operations Calculation ----------")

    return {"wdps_num_executions": calculated_executions}









#with open('./example_codes/step8_example_before.cu', 'r') as file:
#    step8_example_before = file.read()
#with open('./example_codes/step8_example_after.cu', 'r') as file:
#    step8_example_after = file.read()
with open('./example_codes/step8_examples.cu', 'r') as file:
    step8_examples = file.read()

def kernel_num_ops_annotator(state: KernelAnalysisState, config):

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
            "Kernel source code:\n{snippet_kernel_src_concretized_values}\n"
            )
    ])
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
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





def kernel_ops_summarizer(state: KernelAnalysisState, config):

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         #"You are a code summarizer that summarizes the number of integer (INTOP), single-precision (SP-FLOP), and double-precision (DP-FLOP) floating point operations performed by the given C/C++ CUDA kernel source code snippet.\n"
         "You are a code summarizer that summarizes the number of single-precision (SP-FLOP) and double-precision (DP-FLOP) floating point operations performed by the given C/C++ CUDA kernel source code snippet.\n"
         "You will be given two annotated source code snippets: one with warp divergence points annotated with the number of threads that will execute at each part of the kernel code, and another with the number of operations performed at each line of the kernel code.\n"
         "Use the annotated codes to sum up the total number of operations performed by the kernel, accounting for the number of threads that will enter each warp divergence point.\n"
         "The output should briefly explain how it arrived at the total number of SP-FLOP and DP-FLOP operations performed by the kernel.\n"
         "At the end of the summary, return the final summed counts as in the example below:\n"
         "SP-FLOP: YYY\nDP-FLOP: ZZZ\n"
         ),
        ("human",
            "Kernel Invocation Arguments and Descriptions:\n{snippet_first_kernel_invocation}\n\n"
            "Grid Size: {grid_size}\nBlock Size: {block_size}\nTotal Number of Threads: {total_num_threads}\n\n"
            "Kernel source code with SP-FLOP, and DP-FLOP annotations:\n{kernel_annotated_num_ops}\n"
            "Kernel source code with warp divergence and thread count annotations:\n{kernel_annotated_num_threads}\n\n"
            )
    ])
    chain = prompt | llm.with_config(configurable=config.get("configurable", {}))
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
