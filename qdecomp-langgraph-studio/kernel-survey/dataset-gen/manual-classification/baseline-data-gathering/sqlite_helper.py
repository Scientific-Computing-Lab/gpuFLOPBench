import sqlite3
import os

def get_thread_ids_from_sqlite(full_path: str, success_only: bool = False) -> list[str]:
    # open the sqlite file and read the column of thread_id

    # check that the file exists before trying to open it
    if not os.path.exists(full_path):
        return []

    conn = sqlite3.connect(full_path)
    cursor = conn.cursor()

    # check that the checkpoints table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';")
    if cursor.fetchone() is None:
        conn.close()
        return []
    
    # there are duplicate thead_ids, we want the unique ones
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
    thread_ids = cursor.fetchall()

    unique_thread_ids = [str(id[0]) for id in thread_ids]

    conn.close()
    return unique_thread_ids


def split_thread_id_to_parts(thread_id: str) -> dict[str, str]:
    # split the thread_id into model_name, prompt_type, provider_name
    parts = thread_id.split(':')
    if len(parts) != 10:
        raise ValueError(f"Invalid thread_id format: {thread_id}")

    # the different parts are:
    # combined_name, model name, provider url, trial number, prompt type, variant type, nnz_flop_state, top_p, temp
    to_return = {
        'combined_name': parts[0],
        'model_name': parts[1],
        'provider': parts[2] + parts[3],
        'trial_number': parts[4],
        'prompt_type': parts[5],
        'variant_type': parts[6],
        'nnz_flop_state': parts[7],
        'top_p': parts[8],
        'temp': parts[9],
    }
    return to_return