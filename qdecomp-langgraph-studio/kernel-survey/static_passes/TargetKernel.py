import json
from tree_sitter import Language, Parser
import tree_sitter_cuda

CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
parser = Parser(CUDA_LANGUAGE)


class TargetKernel():
    has_float_division: bool
    float_division_line_num : int

    has_cuda_library_function_calls: bool
    cuda_library_function_call_line_num : int

    has_recursion: bool
    recursion_line_num : int

    has_warp_divergence: bool
    warp_divergence_line_num : int

    has_data_dependent_warp_divergence: bool
    has_data_dependent_warp_divergence_line_num : int

    has_common_subexpression: bool
    has_common_subexpression_line_num : int

    has_special_math_function: bool
    special_math_function_line_num : int

    source_code: str
    tree: object
    root_node: object

    def __init__(self, source_code: str):
        self.has_float_division = False
        self.float_division_line_num = -1
        self.has_cuda_library_function_calls = False
        self.cuda_library_function_call_line_num = -1
        self.has_recursion = False
        self.recursion_line_num = -1
        self.has_warp_divergence = False
        self.warp_divergence_line_num = -1
        self.has_data_dependent_warp_divergence = False
        self.has_data_dependent_warp_divergence_line_num = -1
        self.has_common_subexpression = False
        self.common_subexpression_line_num = -1
        self.has_special_math_function = False
        self.special_math_function_line_num = -1

        self.source_code = source_code

        # Use the treesitter parser to analyze the kernel source code
        self.tree = parser.parse(bytes(source_code, "utf8"))
        self.root_node = self.tree.root_node

    def to_dict(self):
        return {
            "has_float_division": self.has_float_division,
            "float_division_line_num": self.float_division_line_num,
            "has_cuda_library_function_calls": self.has_cuda_library_function_calls,
            "cuda_library_function_call_line_num": self.cuda_library_function_call_line_num,
            "has_recursion": self.has_recursion,
            "recursion_line_num": self.recursion_line_num,
            "has_warp_divergence": self.has_warp_divergence,
            "warp_divergence_line_num": self.warp_divergence_line_num,
            "has_data_dependent_warp_divergence": self.has_data_dependent_warp_divergence,
            "has_data_dependent_warp_divergence_line_num": self.has_data_dependent_warp_divergence_line_num,
            "has_common_subexpression": self.has_common_subexpression,
            "common_subexpression_line_num": self.common_subexpression_line_num,
            "has_special_math_function": self.has_special_math_function,
            "special_math_function_line_num": self.special_math_function_line_num,
            "source_code": self.source_code,
        }


class TargetKernelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TargetKernel):
            return obj.to_dict()
        return super().default(obj)
