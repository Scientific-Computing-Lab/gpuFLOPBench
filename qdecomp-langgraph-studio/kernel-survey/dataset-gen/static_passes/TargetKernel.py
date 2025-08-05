import json
from tree_sitter import Language, Parser
import tree_sitter_cuda
from typing import List
from utils.ts_helper import get_function_name, find_function_calls, find_all_declaration_nodes

CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
parser = Parser(CUDA_LANGUAGE)


class TargetKernel():
    @staticmethod
    def get_global_func_name(source_code: str) -> str:
        """Extracts the name of the __global__ kernel function from the source code."""
        tree = parser.parse(bytes(source_code, "utf8"))
        root = tree.root_node
        # Find all function declaration nodes
        decl_nodes = find_all_declaration_nodes(root)
        global_funcs = []
        for decl in decl_nodes:
            # Determine the function_definition node
            func_def = decl if decl.type == 'function_definition' else next((c for c in decl.children if c.type == 'function_definition'), None)
            if not func_def:
                continue
            # Collect specifier nodes preceding the declarator or body
            spec_nodes = []
            for child in func_def.children:
                if child.type in ('function_declarator', 'compound_statement'):
                    break
                spec_nodes.append(child)
            # Check for __global__ specifier
            queue = spec_nodes.copy()
            is_global = False
            while queue:
                curr = queue.pop(0)
                if curr.type == '__global__':
                    is_global = True
                    break
                queue.extend(curr.children)
            if is_global:
                name = get_function_name(decl)
                if name:
                    global_funcs.append(name)
        # No global function found
        if not global_funcs:
            return None
        # Single global function
        if len(global_funcs) == 1:
            return global_funcs[0]
        # Multiple globals: prefer the caller (not called by another global)
        called = set()
        find_function_calls(root, called)
        candidates = [name for name in global_funcs if name not in called]
        if candidates:
            return candidates[0]
        # Fallback to first
        return global_funcs[0]


    has_float_division: bool
    float_division_line_num : List[int]

    has_external_function_calls: bool
    external_function_call_line_num : List[int]

    has_recursion: bool
    recursion_line_num : List[int]

    has_warp_divergence: bool
    warp_divergence_line_num : List[int]

    has_data_dependent_warp_divergence: bool
    data_dependent_warp_divergence_line_num : List[int]

    has_common_subexpression: bool
    common_subexpression_line_num : List[int]

    has_special_math_function: bool
    special_math_function_line_num : List[int]

    source_code: str
    tree: object
    root_node: object
    global_function_name: str

    def __init__(self, source_code: str):
        self.has_float_division = False
        self.float_division_line_num = []
        self.has_external_function_calls = False
        self.external_function_call_line_num = []
        self.has_recursion = False
        self.recursion_line_num = []
        self.has_warp_divergence = False
        self.warp_divergence_line_num = []
        self.has_data_dependent_warp_divergence = False
        self.data_dependent_warp_divergence_line_num = []
        self.has_common_subexpression = False
        self.common_subexpression_line_num = []
        self.has_special_math_function = False
        self.special_math_function_line_num = []

        self.source_code = source_code
        self.global_function_name = TargetKernel.get_global_func_name(source_code)

        # Use the treesitter parser to analyze the kernel source code
        self.tree = parser.parse(bytes(source_code, "utf8"))
        self.root_node = self.tree.root_node

    def to_dict(self):
        return {
            "has_float_division": self.has_float_division,
            "float_division_line_num": self.float_division_line_num,

            "has_external_function_calls": self.has_external_function_calls,
            "external_function_call_line_num": self.external_function_call_line_num,

            "has_recursion": self.has_recursion,
            "recursion_line_num": self.recursion_line_num,

            "has_warp_divergence": self.has_warp_divergence,
            "warp_divergence_line_num": self.warp_divergence_line_num,

            "has_data_dependent_warp_divergence": self.has_data_dependent_warp_divergence,
            "data_dependent_warp_divergence_line_num": self.data_dependent_warp_divergence_line_num,

            "has_common_subexpression": self.has_common_subexpression,
            "common_subexpression_line_num": self.common_subexpression_line_num,

            "has_special_math_function": self.has_special_math_function,
            "special_math_function_line_num": self.special_math_function_line_num,

            "source_code": self.source_code,
            "global_function_name": self.global_function_name
        }



class TargetKernelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TargetKernel):
            return obj.to_dict()
        return super().default(obj)
