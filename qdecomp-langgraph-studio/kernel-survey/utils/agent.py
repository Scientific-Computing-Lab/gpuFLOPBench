from typing import TypedDict, Literal
from utils.nodes import *
from utils.state import KernelAnalysisState
from utils.configuration import Configuration

from langgraph.graph import StateGraph, END


workflow = StateGraph(KernelAnalysisState, config_schema=Configuration)

workflow.add_node("get_input_problem_0", get_input_problem)

workflow.add_node("float_div_check_1", float_div_check_node)

# Graph edges
workflow.add_edge("get_input_problem_0", "float_div_check_1")

workflow.add_edge("float_div_check_1", END)


# Set entrypoint
workflow.set_entry_point("get_input_problem_0")

graph = workflow.compile()
