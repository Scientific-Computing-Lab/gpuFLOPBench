import re
from tree_sitter import Language, Parser
import tree_sitter_cuda
from copy import deepcopy
from collections import defaultdict, deque

# -------------------------------------------------------------------
# STEP 0: Load the Tree-sitter CUDA parser
# -------------------------------------------------------------------
CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
parser = Parser(CUDA_LANGUAGE)


def parse_combined_sources(combined: str) -> dict:
    # look for lines that consist *only* of 10+ dashes, using ^/$ in MULTILINE mode
    pattern = re.compile(
        r'^(?P<sep>-{10,})[ \t]*$'   # a line of 10+ dashes (maybe trailing spaces)
        r'\r?\n'
        r'(?P<name>.+?)\r?\n'        # the filename line
        r'(?P=sep)[ \t]*$'           # the same exact sep on its own line
        r'\r?\n+'                    # one or more blank lines
        r'(?P<code>.*?)(?='
          r'^(?:-{10,})[ \t]*$'      # lookahead for the next sep
          r'\r?\n'
          r'.+\r?\n'
          r'^(?:-{10,})[ \t]*$'
        r'|\Z)',
        re.DOTALL | re.MULTILINE
    )

    files = {}
    for m in pattern.finditer(combined):
        name = m.group('name').strip()
        code = m.group('code')
        files[name] = code
    return files


def render_combined_sources(sources: dict) -> str:
    sep = '-'*20
    parts = []
    for name, code in sources.items():
        parts += [sep, name, sep+"\n", code.rstrip("\n"), ""]
    return "\n".join(parts).rstrip() + "\n"