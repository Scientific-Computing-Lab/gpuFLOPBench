import re
from tree_sitter import Language, Parser
import tree_sitter_cuda
from copy import deepcopy

# -------------------------------------------------------------------
# STEP 0: Load the Tree-sitter CUDA parser
# -------------------------------------------------------------------
CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
parser = Parser(CUDA_LANGUAGE)

# -------------------------------------------------------------------
# A simple text-replacement infrastructure based on Tree-sitter byte offsets.
# -------------------------------------------------------------------
class Rewriter:
    def __init__(self, source_bytes: bytes):
        self.source_bytes = source_bytes
        self.edits = []

    def replace(self, start_byte: int, end_byte: int, new_text: str):
        self.edits.append((start_byte, end_byte, new_text.encode('utf8')))

    def finish(self) -> bytes:
        src = self.source_bytes
        for start, end, txt in sorted(self.edits, key=lambda e: -e[0]):
            src = src[:start] + txt + src[end:]
        return src

# -------------------------------------------------------------------
# AST traversal for initial argv and variable value collection and replacement
# -------------------------------------------------------------------
def collect_and_rewrite(tree, source_bytes, argv_values, rewriter):
    var_map = {}
    cursor = tree.walk()

    def node_text(n):
        return source_bytes[n.start_byte:n.end_byte].decode('utf8')

    def traverse(n):
        # 1) Replace argv[N] with literal
        if n.type == 'subscript_expression':
            c0, c1 = n.children[0], n.children[1]
            if c0.type == 'identifier' and node_text(c0) == 'argv' \
               and c1.type == 'subscript_argument_list':
                if len(c1.children) > 1:
                    idx_node = c1.children[1]
                    if idx_node.type == 'number_literal':
                        idx = int(node_text(idx_node))
                        if idx < len(argv_values):
                            lit = argv_values[idx]
                            cpp_lit = '"' + lit.replace('"', '\\"') + '"'
                            rewriter.replace(n.start_byte, n.end_byte, cpp_lit)
                            return

        # 1a) Replace atoi(argv[N]) with integer literal (if possible)
        if n.type == 'call_expression':
            if len(n.children) >= 2:
                call_ident = n.children[0]
                arglist = n.children[1]
                if call_ident.type == 'identifier' and node_text(call_ident) == 'atoi':
                    # Find the argument node, look for argv[N]
                    for arg in arglist.named_children:
                        if arg.type == 'subscript_expression':
                            c0, c1 = arg.children[0], arg.children[1]
                            if c0.type == 'identifier' and node_text(c0) == 'argv' \
                               and c1.type == 'subscript_argument_list':
                                if len(c1.children) > 1:
                                    idx_node = c1.children[1]
                                    if idx_node.type == 'number_literal':
                                        idx = int(node_text(idx_node))
                                        if idx < len(argv_values):
                                            lit = argv_values[idx]
                                            try:
                                                int_lit = str(int(lit))
                                            except Exception:
                                                int_lit = '0'
                                            rewriter.replace(n.start_byte, n.end_byte, int_lit)
                                            return

        # 2) Track simple assignments or declarations:
        #    var = "literal";   var = other_var;
        if n.type in ('variable_declaration', 'assignment_expression'):
            id_node = None
            eq_idx = None
            for i, c in enumerate(n.children):
                if c.type == 'identifier' and id_node is None:
                    id_node = c
                if c.type == '=' and eq_idx is None:
                    eq_idx = i
            if id_node and eq_idx is not None and eq_idx+1 < len(n.children):
                rhs = n.children[eq_idx+1]
                rhs_txt = node_text(rhs).strip()
                # literal string
                if rhs.type == 'string_literal':
                    var_map[node_text(id_node)] = rhs_txt
                # numeric literal
                elif rhs.type == 'number_literal':
                    var_map[node_text(id_node)] = rhs_txt
                # RHS is an identifier we already know
                elif rhs.type == 'identifier' and rhs_txt in var_map:
                    var_map[node_text(id_node)] = var_map[rhs_txt]
                    start, end = rhs.start_byte, rhs.end_byte
                    rewriter.replace(start, end, var_map[rhs_txt])

        for c in n.children:
            traverse(c)

    traverse(cursor.node)
    return var_map

# -------------------------------------------------------------------
# Propagate hard-coded values into function call argument_list
# -------------------------------------------------------------------
def propagate_literals_to_calls(tree, source_bytes, var_map, rewriter):
    def node_text(n):
        return source_bytes[n.start_byte:n.end_byte].decode('utf8')

    def traverse(n):
        if n.type == 'call_expression':
            # Find argument_list node (tree-sitter-cpp or cuda: may be named 'argument_list')
            for child in n.children:
                if child.type == 'argument_list':
                    for arg in child.named_children:
                        # If argument is an identifier and its value is a hard-coded literal,
                        # replace it in the source with the literal.
                        if arg.type == 'identifier':
                            name = node_text(arg)
                            if name in var_map:
                                value = var_map[name]
                                rewriter.replace(arg.start_byte, arg.end_byte, value)
        for c in n.children:
            traverse(c)

    traverse(tree.root_node)

# -------------------------------------------------------------------
# Split and re-combine multi-file string format
# -------------------------------------------------------------------
def parse_combined_sources(combined: str) -> dict:
    pattern = re.compile(
        r'(?P<sep>-{10,})\n'
        r'(?P<name>.+)\n'
        r'(?P=sep)\n+'
        r'(?P<code>.*?)(?=(?:-{10,}\n.+\n-{10,})|\Z)',
        re.DOTALL
    )
    files = {}
    for m in pattern.finditer(combined):
        name = m.group('name').strip()
        code = m.group('code')
        files[name] = code
    return files

def render_combined_sources(sources: dict) -> str:
    sep = '-' * 18
    parts = []
    for name, code in sources.items():
        parts.append(sep)
        parts.append(name)
        parts.append(sep + '\n')
        parts.append(code.rstrip('\n'))
        parts.append('')
    return '\n'.join(parts).rstrip() + '\n'

# -------------------------------------------------------------------
# Perform iterative rewriting until no more changes
# -------------------------------------------------------------------
def rewrite_until_fixed(src: str, argv: list) -> str:
    src_bytes = bytes(src, 'utf8')
    while True:
        rewriter = Rewriter(src_bytes)
        tree = parser.parse(src_bytes)
        var_map = collect_and_rewrite(tree, src_bytes, argv, rewriter)
        # After collecting, propagate hard-coded variables to call argument_lists
        propagate_literals_to_calls(tree, src_bytes, var_map, rewriter)
        updated = rewriter.finish()
        if updated == src_bytes:
            return src_bytes.decode('utf8')
        src_bytes = updated

# -------------------------------------------------------------------
# Main propagation over combined multi-file string
# -------------------------------------------------------------------
def propagate_argv_in_combined(combined: str, argv: list) -> str:
    srcs = parse_combined_sources(combined)
    new_srcs = {}
    for fname, src in srcs.items():
        new_srcs[fname] = rewrite_until_fixed(src, argv)
    return render_combined_sources(new_srcs)

# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
# if __name__ == '__main__':
#     combined_code = '''
# ------------------
# main.cu
# ------------------
# 
# #include <iostream>
# #include <string>
# #include "helper.cuh"
# 
# int main(int argc, char** argv) {
#     auto a = argv[1];
#     std::string b = a;
#     helper::print_upper(b);
#     return 0;
# }
# 
# ------------------
# helper.cu
# ------------------
# 
# #include "helper.cuh"
# #include <algorithm>
# #include <iostream>
# 
# namespace helper {
#     void print_upper(const std::string &s) {
#         std::string tmp = s;
#         std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::toupper);
#         std::cout << tmp << std::endl;
#     }
# }
# 
# ------------------
# helper.cuh
# ------------------
# 
# #pragma once
# #include <string>
# namespace helper {
#     void print_upper(const std::string &s);
# }
# '''.lstrip()
# 
# argv_values = ['prog', 'Iterate!']
# print(propagate_argv_in_combined(combined_code, argv_values))