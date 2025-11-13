import streamlit as st
import sys
from typing import List, Dict, Any
import os
import json
import streamlit.components.v1 as components

sys.path.append("..")  # Adjust the path to import from the parent directory
from utils.dataset import kernels_data  # noqa: E402

# ---------------- Configuration -----------------
# Define the checklist categories to apply to each kernel snippet.
CHECKLIST_CATEGORIES: List[str] = [
    "1: Has Warp Divergence",
    "2: Has Data-Dependent Warp Divergence",
    "3: Has FLOP Division",
    "4: Calls EXTERNAL or LIB function",
    "5: Calls DEVICE function",
    "6: Calls special math function",
    "7: Has Common Subexpression",
    "8: Has Recursion",
    "9: Missing Some Code",
]

SAVE_FILE_NAME = "manually_classified_CUDA_kernels.json"
SAVE_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), SAVE_FILE_NAME))

# -------------- Helper Functions ---------------

def flatten_kernels(data: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, Any]]:
    """Flatten the nested kernels_data structure into a list.

    Each entry: { 'project': str, 'file': str, 'snippet_index': int, 'code': str, 'id': str }
    """
    flat = []
    for project, files in data.items():
        for file_path, snippets in files.items():
            for i, code in enumerate(snippets):
                uid = f"{project}|{file_path}|{i}"
                flat.append({
                    "project": project,
                    "file": file_path,
                    "snippet_index": i,
                    "code": code,
                    "id": uid,
                })
    return flat


def load_saved_progress():
    if os.path.exists(SAVE_FILE_PATH):
        try:
            with open(SAVE_FILE_PATH, "r") as f:
                data = json.load(f)
            saved_categories = data.get("categories", [])
            # Only load classifications if categories match (simple check)
            if saved_categories == CHECKLIST_CATEGORIES:
                return data.get("classifications", {}), data.get("last_index", 0)
            return {}, 0
        except Exception:
            return {}, 0
    return {}, 0


def save_progress():
    payload = {
        "categories": CHECKLIST_CATEGORIES,
        "classifications": st.session_state.classifications,
        # Persist "farthest" index visited instead of current only
        "last_index": st.session_state.get("max_index_reached", st.session_state.current_idx),
        "total_snippets": len(st.session_state.flat_kernels),
    }
    try:
        with open(SAVE_FILE_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        st.toast(f"Failed to save progress: {e}", icon="⚠️")


def init_state():
    if "flat_kernels" not in st.session_state:
        st.session_state.flat_kernels = flatten_kernels(kernels_data)
    if "classifications" not in st.session_state:
        classifications, last_index = load_saved_progress()
        st.session_state.classifications = classifications  # id -> {category: bool}
        st.session_state.current_idx = min(last_index, len(st.session_state.flat_kernels) - 1) if st.session_state.flat_kernels else 0
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "visited" not in st.session_state:
        st.session_state.visited = set(st.session_state.classifications.keys())
    if "max_index_reached" not in st.session_state:
        # Initialize with persisted last_index or current
        st.session_state.max_index_reached = st.session_state.current_idx


def set_index(new_idx: int):
    total = len(st.session_state.flat_kernels)
    if 0 <= new_idx < total:
        st.session_state.current_idx = new_idx
        st.session_state.max_index_reached = max(st.session_state.max_index_reached, new_idx)

# --- Navigation button callbacks ---

def go_prev():
    if st.session_state.current_idx > 0:
        st.session_state.current_idx -= 1
        # No update to max needed (going backwards)

def go_next():
    total = len(st.session_state.flat_kernels)
    if st.session_state.current_idx < total - 1:
        save_progress()  # persist before moving forward
        st.session_state.current_idx += 1
        st.session_state.max_index_reached = max(st.session_state.max_index_reached, st.session_state.current_idx)

def jump_latest():
    # Jump to farthest index reached so far
    target = st.session_state.max_index_reached
    total = len(st.session_state.flat_kernels)
    if 0 <= target < total:
        st.session_state.current_idx = target

# -------------- UI Rendering --------------------

def render_navigation():
    total = len(st.session_state.flat_kernels)
    visited = len(st.session_state.visited)
    progress_pct = (visited / total * 100.0) if total else 0

    left, mid, right = st.columns([1, 2, 1])
    with left:
        st.button(
            "Previous",
            key="btn_prev",
            disabled=st.session_state.current_idx <= 0,
            on_click=go_prev,
        )
    with mid:
        st.markdown(
            f"### Manual Classification — Progress: {progress_pct:.1f}% ({visited} / {total})"
        )
        st.caption(
            f"Viewing snippet {st.session_state.current_idx + 1} of {total}" if total else "No snippets loaded"
        )
    with right:
        st.button(
            "Next",
            key="btn_next",
            disabled=st.session_state.current_idx >= total - 1,
            on_click=go_next,
        )
        st.button(
            "Jump to Latest",
            key="btn_jump_latest",
            disabled=st.session_state.current_idx == st.session_state.max_index_reached,
            on_click=jump_latest,
            help=f"Go to farthest visited index ({st.session_state.max_index_reached + 1})" if total else None,
        )


def render_current_snippet():
    if not st.session_state.flat_kernels:
        st.warning("No kernel snippets available. Ensure 'extracted_CUDA_kernels.json' exists and is valid.")
        return

    entry = st.session_state.flat_kernels[st.session_state.current_idx]
    snippet_id = entry["id"]
    st.session_state.visited.add(snippet_id)
    # Update max index reached when viewing (covers direct jumps)
    st.session_state.max_index_reached = max(st.session_state.max_index_reached, st.session_state.current_idx)

    # Container for snippet classification + code
    with st.container(border=True):  # Streamlit >=1.32 supports border arg; safe to ignore if older
        header_cols = st.columns([3, 2])
        with header_cols[0]:
            st.subheader(f"Project: {entry['project']}")
            st.caption(entry["file"])
        with header_cols[1]:
            st.caption(f"Snippet Index in File: {entry['snippet_index']}")

        # Checklist moved to sidebar; show code full width here
        st.code(body=entry["code"], language='cpp', line_numbers=True)


def render_sidebar():
    with st.sidebar:
        # Separator before checklist and summary as requested
        st.divider()
        st.subheader("Checklist")
        if st.session_state.flat_kernels:
            entry = st.session_state.flat_kernels[st.session_state.current_idx]
            snippet_id = entry["id"]
            current_values = st.session_state.classifications.get(snippet_id, {})
            updated = {}
            for cat in CHECKLIST_CATEGORIES:
                key = f"chk::{snippet_id}::{cat}"
                updated[cat] = st.checkbox(cat, value=current_values.get(cat, False), key=key)
            st.session_state.classifications[snippet_id] = updated
        st.divider()
        st.header("Summary")
        total = len(st.session_state.flat_kernels)
        classified = sum(
            1 for v in st.session_state.classifications.values() if any(v.values())
        )
        st.metric("Snippets Classified", f"{classified}/{total}")
        if st.button("Export JSON"):
            export_data = {
                "categories": CHECKLIST_CATEGORIES,
                "classifications": st.session_state.classifications,
                "last_index": st.session_state.get("max_index_reached", st.session_state.current_idx),
            }
            st.download_button(
                label="Download Results",
                file_name=SAVE_FILE_NAME,
                mime="application/json",
                data=json.dumps(export_data, indent=2),
            )
        if st.button("Save Now"):
            save_progress()
            st.success("Progress saved.")


def inject_keyboard_shortcuts():
    """Attach keyboard shortcuts by loading a tiny components.html iframe that
    registers a single global listener on the parent document (the main app).

    Runs every rerun to ensure listener survives DOM refreshes. A guard on
    window.parent (__kernelKeysInstalled) prevents duplicate handlers.

    Keys:
      Left / A  -> Previous
      Right / D -> Next
      0        -> Jump to Latest
      1-9      -> Toggle checklist categories 1..9
    """
    categories_json = json.dumps(CHECKLIST_CATEGORIES)
    # Use a plain triple-quoted string and replace placeholder to avoid f-string brace issues
    iframe_html = """
    <html><head><meta charset='utf-8'></head><body style='margin:0;padding:0;'>
    <script>
    (function() {
      try {
        const categories = __CATEGORIES__;
        if (window.parent && !window.parent.__kernelKeysInstalled) {
          window.parent.__kernelKeysInstalled = true;
          const keyBlockTags = ['INPUT','TEXTAREA'];
          // Build digit->category label mapping (1-based)
          const numberToLabel = {};
          categories.forEach((c,i)=>{ numberToLabel[(i+1).toString()] = c; });
          function clickButtonWithText(txt) {
            const doc = window.parent.document;
            const btns = Array.from(doc.querySelectorAll('button'));
            const b = btns.find(el => el.innerText.trim() === txt);
            if (b) b.click();
          }
          function toggleCheckboxWithText(labelText) {
            if (!labelText) return;
            const doc = window.parent.document;
            // Find exact matching label (trimmed innerText)
            const labels = Array.from(doc.querySelectorAll('label'));
            let target = labels.find(l => l.innerText.trim() === labelText);
            if (!target) {
              // Fallback: find span exact text then ascend
              const spans = Array.from(doc.querySelectorAll('span'));
              const span = spans.find(s => s.innerText.trim() === labelText);
              if (span) target = span.closest('label');
            }
            if (target) {
              const input = target.querySelector('input[type=\"checkbox\"]');
              if (input) {
                try {
                  const evOpts = {bubbles:true,cancelable:true};
                  input.dispatchEvent(new MouseEvent('pointerdown', evOpts));
                  input.dispatchEvent(new MouseEvent('mousedown', evOpts));
                  input.click();
                  input.dispatchEvent(new MouseEvent('mouseup', evOpts));
                  input.dispatchEvent(new MouseEvent('pointerup', evOpts));
                } catch(e) { console.warn('toggleCheckboxWithText error', e); }
              }
            }
          }
          let lastKeyTime = 0;
          window.parent.addEventListener('keydown', function(e) {
            try { console.log('[KernelClassify] keydown:', e.key, e.code); } catch(_) {}
            if (keyBlockTags.includes(e.target.tagName)) return;
            const now = Date.now();
            if (now - lastKeyTime < 50) return; // debounce
            let k = e.key;
            if (/^Numpad[0-9]$/.test(e.code)) k = e.code.replace('Numpad','');
            if (k === 'ArrowLeft' || k === 'a' || k === 'A') {
              clickButtonWithText('Previous');
            } else if (k === 'ArrowRight' || k === 'd' || k === 'D' || k === 'Enter') {
              clickButtonWithText('Next');
            } else if (k === '0') {
              clickButtonWithText('Jump to Latest');
            } else if (numberToLabel[k]) {
              toggleCheckboxWithText(numberToLabel[k]);
            } else {
              return;
            }
            lastKeyTime = now;
          }, true);
          console.log('[KernelClassify] keyboard shortcuts installed');
        } else {
          // Already installed; no action needed.
        }
      } catch(e) {
        console.warn('Shortcut injection error', e);
      }
    })();
    </script>
    </body></html>
    """.replace("__CATEGORIES__", categories_json)
    components.html(iframe_html, height=1, width=1)

# ------------------ Main App --------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Manual Kernel Classification", layout="wide")
    init_state()
    render_navigation()
    st.divider()
    render_current_snippet()
    render_sidebar()
    inject_keyboard_shortcuts()