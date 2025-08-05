import sys
import os
from tree_sitter import Language, Parser
import tree_sitter_cuda

# Re-use the tree-sitter setup from extract_kernels.py
CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
parser = Parser(CUDA_LANGUAGE)

def print_tree(node, source_lines, printed_lines, level=0):
    """
    Recursively prints the AST, showing node types, line numbers, and decoded text for each node.
    It also prints the full source line before detailing the nodes on it.
    """
    indent = "  " * level
    start_line, start_col = node.start_point
    end_line, end_col = node.end_point

    # Print the source line if it hasn't been printed yet.
    if start_line not in printed_lines:
        print(f"\n// Source Line {start_line + 1}: {source_lines[start_line]}")
        printed_lines.add(start_line)

    # Decode the node's text and show the first line for multi-line nodes
    node_text_lines = node.text.decode('utf8').split('\\n')
    first_line = node_text_lines[0]
    
    # Truncate long lines for readability
    if len(first_line) > 70:
        first_line = first_line[:67] + "..."

    print(f"{indent}L{start_line + 1}:{start_col}-L{end_line + 1}:{end_col} [{node.type}] '{first_line}'")

    for child in node.children:
        print_tree(child, source_lines, printed_lines, level + 1)

def main():
    """
    Main function to parse a file provided via command line and print its AST.
    """
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <filename>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_code = f.read()
        
        tree = parser.parse(bytes(source_code, "utf8"))
        source_lines = source_code.splitlines()
        
        print(f"--- Tree-sitter AST for {file_path} ---")
        print_tree(tree.root_node, source_lines, set())
        print("--- End of AST ---")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
