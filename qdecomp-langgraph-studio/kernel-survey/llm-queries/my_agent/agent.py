from typing import TypedDict, Literal
from my_agent.utils.nodes import *
from my_agent.utils.state import KernelAnalysisState
from my_agent.utils.configuration import Configuration

from langgraph.graph import StateGraph, END

# please create a file called .openrouter-api-key in the current directory
#with open('./.openrouter-api-key', 'r') as file:
#    OPENROUTER_API_KEY=file.read().strip()
#    #os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY


workflow = StateGraph(KernelAnalysisState, config_schema=Configuration)

workflow.add_node("get_input_problem_0", get_input_problem)

workflow.add_node("src_input_args_concretizer_1", src_input_args_concretizer)
workflow.add_node("concretization_checker_1a", concretization_checker)

workflow.add_node("src_single_kernel_execution_modifier_2", src_single_kernel_execution_modifier)
workflow.add_node("single_kernel_execution_checker_2a", single_kernel_execution_checker)

workflow.add_node("first_kernel_invocation_snippet_extractor_3", first_kernel_invocation_snippet_extractor)
workflow.add_node("kernel_source_snippet_extractor_4", kernel_source_snippet_extractor)

workflow.add_node("kernel_source_snippet_concretizer_5", kernel_source_snippet_concretizer)
workflow.add_node("snippet_concretization_checker_5a", snippet_concretization_checker)

workflow.add_node("kernel_warp_divergence_annotator_6", kernel_warp_divergence_annotator)

workflow.add_node("kernel_wdp_variables_annotator_7", kernel_wdp_variables_annotator)
workflow.add_node("wdp_list_extractor_7a", wdp_extractor)
workflow.add_node("wdp_num_execution_calculations_7b", wdp_num_executions_calculations)

workflow.add_node("kernel_num_ops_annotator_8", kernel_num_ops_annotator)
workflow.add_node("num_ops_checker_8a", num_ops_checker)



workflow.add_node("kernel_ops_summarizer_9", kernel_ops_summarizer)


# Graph edges
workflow.add_edge("get_input_problem_0", "src_input_args_concretizer_1")

workflow.add_edge("src_input_args_concretizer_1", "concretization_checker_1a")

workflow.add_conditional_edges(source="concretization_checker_1a", 
                               path=route_concretization_status_edge,
                               path_map={
                                      "ACCEPT": "src_single_kernel_execution_modifier_2",
                                      "REJECT": "src_input_args_concretizer_1"
                               },)

#workflow.add_edge("src_input_args_concretizer_1", "src_single_kernel_execution_modifier_2")

#workflow.add_edge("src_single_kernel_execution_modifier_2", "first_kernel_invocation_snippet_extractor_3")

workflow.add_edge("src_single_kernel_execution_modifier_2", "single_kernel_execution_checker_2a")

workflow.add_conditional_edges(source="single_kernel_execution_checker_2a", 
                               path=route_single_kernel_source_status_edge,
                               path_map={
                                      "ACCEPT": "first_kernel_invocation_snippet_extractor_3",
                                      "REJECT": "src_single_kernel_execution_modifier_2"
                               },)

workflow.add_edge("first_kernel_invocation_snippet_extractor_3", "kernel_source_snippet_extractor_4")

workflow.add_edge([
    "kernel_source_snippet_extractor_4",
    #"first_kernel_invocation_snippet_extractor_3"
    ],  "kernel_source_snippet_concretizer_5")

workflow.add_edge("kernel_source_snippet_concretizer_5", "snippet_concretization_checker_5a")

workflow.add_conditional_edges(source="snippet_concretization_checker_5a", 
                               path=route_snippet_concretization_status_edge,
                               path_map={
                                      "ACCEPT": "kernel_warp_divergence_annotator_6",
                                      "REJECT": "kernel_source_snippet_concretizer_5"
                               },)

#workflow.add_edge("snippet_concretization_checker_5a", "kernel_warp_divergence_annotator_6")

workflow.add_edge([
    "kernel_warp_divergence_annotator_6",
#    #"snippet_concretization_checker_5a"
    ],  "kernel_wdp_variables_annotator_7")


workflow.add_edge([
    #"first_kernel_invocation_snippet_extractor_3",
    "kernel_warp_divergence_annotator_6"
    ],  "kernel_num_ops_annotator_8")

workflow.add_edge("kernel_num_ops_annotator_8", "num_ops_checker_8a")

workflow.add_conditional_edges(source="num_ops_checker_8a", 
                               path=route_num_ops_annotation_status_edge,
                               path_map={
                                      "ACCEPT": "wdp_num_execution_calculations_7b",
                                      #"ACCEPT": "kernel_wdp_variables_annotator_7",
                                      "REJECT": "kernel_num_ops_annotator_8"
                               },)

workflow.add_edge("kernel_wdp_variables_annotator_7", "wdp_list_extractor_7a")

workflow.add_edge("wdp_list_extractor_7a", "wdp_num_execution_calculations_7b")

workflow.add_conditional_edges(
    source="wdp_num_execution_calculations_7b",
    path=wdp_num_executions_looper,
    path_map={
        "continue": "wdp_num_execution_calculations_7b",
        "end": "kernel_ops_summarizer_9"
    }
)

workflow.add_edge("kernel_ops_summarizer_9", END)

# Set entrypoint
workflow.set_entry_point("get_input_problem_0")

graph = workflow.compile()
