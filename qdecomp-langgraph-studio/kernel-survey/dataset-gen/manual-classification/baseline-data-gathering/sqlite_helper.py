import sqlite3
import os
import pandas as pd
from agent import make_graph

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


def sqlitefile_to_dataframe(full_path: str):
    # open the sqlite file and read the checkpoints table into a pandas dataframe
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")


    thread_ids = get_thread_ids_from_sqlite(full_path)
    graph = make_graph(full_path)

    df = pd.DataFrame()

    for thread_id in thread_ids:
        parts = split_thread_id_to_parts(thread_id)
        
        parts['langgraph_thread_id'] = thread_id

        # get the state for the thread_id
        config = {'configurable': {'thread_id': thread_id}}
        state = graph.get_state(config)

        full_final_state_dict = parts | state.values

        state_keys = list(state.values.keys())

        # if there is no next state and no tasks left, then we had a successful execution
        if (not state.next) and (not state.tasks):
            #full_final_state_dict['error'] = 'Success'
            full_final_state_dict['state_of_failure'] = None
            assert 'raw_flop_counts' in state_keys, f"raw_flop_counts not in state keys: {state_keys} -- this was supposed to be a successful run..."

        else:
            # we had an error, it should be written in the tasks error
            failure_task = state.tasks[0]
            full_final_state_dict['state_of_failure'] = failure_task.name

        #print("thread_id:", thread_id)
        dict_df = pd.DataFrame([full_final_state_dict])
        df = pd.concat([df, dict_df], ignore_index=True)

    
    df['generic_model_name'] = df['model_name'].apply(lambda x: x.split('/')[1] if '/' in x else x)

    def parse_total_cost(cost_list):
        if cost_list is None or len(cost_list) == 0:
            return None
        #assert len(cost_list) == 1, f"Expected a single cost value, got: {cost_list}"
        # just take the last value, it means the query was re-run for some reason
        return cost_list[-1]

    df['total_cost'] = df['total_cost'].apply(parse_total_cost)

    def parse_total_query_time(time):
        if time is None:
            return None
        if type(time) == list:
            if len(time) == 0:
                return None
            return float(time[-1])
        return float(time)

    df['total_query_time'] = df['total_query_time'].apply(parse_total_query_time)

    def parse_error(x):
        if type(x) == list and len(x) == 0:
            return "Cancelled\nMid-Execution"

        if type(x) == list:
            error = str(x[-1])
            if "JSONDecodeError" in error or 'Expecting value: line ' in error:
                return "JSON\nDecode\nError"
            elif "Invalid JSON" in error:
                return "Invalid\nJSON"
            elif "NoneType" in error:
                return "NoneType\nReturned"
            elif "TIMEOUT" in error:
                return "Query\nTimeout"
            else:
                return error

        return str(x)

    df['error'] = df['error'].apply(parse_error)

    df['has_nz_flops'] = df['nnz_flop_state'].apply(lambda x: 'No' if x == 'Zero SP + DP FLOP' else 'Yes')

    # percent difference, we add a small epsilon to avoid division by zero
    df['percent_diff_sp'] = 100*(df['predicted_sp_flop_count'] - df['empirical_sp_flop_count']) / (df['empirical_sp_flop_count'] + 1e-9)
    df['percent_diff_dp'] = 100*(df['predicted_dp_flop_count'] - df['empirical_dp_flop_count']) / (df['empirical_dp_flop_count'] + 1e-9)

    return df