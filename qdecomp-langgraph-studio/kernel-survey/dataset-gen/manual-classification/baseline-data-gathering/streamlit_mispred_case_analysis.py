
import streamlit as st
from sqlite_helper import sqlitefile_to_dataframe
import pandas as pd
import os
import pathlib
from typing import List

classes_file = os.path.join(os.path.dirname(__file__), 'manualErrorClasses.txt')
classes_file = os.path.abspath(classes_file)

percent_cutoffs = [1, 3, 5, 10, 20, 30, 50, 100]

def get_sqlite_files():
    db_dir = './checkpoints/'

    # get all the sqlite files in the db_dir
    sqlite_files = [f for f in os.listdir(db_dir) if f.endswith('.sqlite')]

    for sqlite_file in sqlite_files:
        print(f"Processing [{sqlite_file}]") 

    # keep only the ones with gpt-5-mini in the name
    sqlite_files = [f for f in sqlite_files if 'gpt-5-mini' in f]
    print(sqlite_files)

    sqlite_files = [os.path.join(db_dir, f) for f in sqlite_files]
    return sqlite_files



sqlite_files = get_sqlite_files()




def load_dataframes(full_paths: list[str]):

    df = pd.DataFrame()

    for sqlite_file in full_paths:
        filedf = sqlitefile_to_dataframe(sqlite_file)
        df = pd.concat([df, filedf], ignore_index=True)

    return df


def _saved_csv_path_for_sqlite(sqlite_path: str) -> str:
    # name the csv after the sqlite file (without directory), replace extension with .csv
    p = pathlib.Path(sqlite_path)
    csv_name = p.stem + '-errorAnalysis.csv'
    # store in same checkpoints directory as sqlite by default
    return str(p.with_name(csv_name))


def save_dataframe(sqlite_path: str, df: pd.DataFrame) -> None:
    """Save dataframe to a CSV named after the sqlite file using quotechar '"'."""
    csv_path = _saved_csv_path_for_sqlite(sqlite_path)
    # ensure directory exists
    csv_dir = os.path.dirname(csv_path) or '.'
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_path, index=False, quotechar='"')


def load_saved_dataframe(sqlite_path: str) -> pd.DataFrame | None:
    """Load previously saved dataframe for this sqlite file if present, else None."""
    csv_path = _saved_csv_path_for_sqlite(sqlite_path)
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path, quotechar='"')
        except Exception:
            # if reading fails, ignore and return None so we fall back to reading sqlite
            return None
    return None

def get_data_by_cutoff(df: pd.DataFrame, percent_cutoff: int):
    # filter the dataframe to only include rows where the absolute value of percent_diff_sp is greater than the cutoff
    filtered_df = df[(abs(df['percent_diff_sp']) > percent_cutoff) | (abs(df['percent_diff_dp']) > percent_cutoff)]
    return filtered_df



def _read_manual_classes_file(path: str) -> List[str]:
    """Read manual error classes from a file, one per line. Returns stripped, non-empty lines.

    If the file doesn't exist, returns an empty list.
    """
    try:
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines()]
        return [ln for ln in lines if ln != '']
    except Exception:
        return []


def _append_manual_class_to_file(path: str, cls: str) -> None:
    """Append a new class to the manual classes file if it's not already present.

    This is safe to call repeatedly; duplicates are avoided by checking existing contents.
    """
    try:
        existing = _read_manual_classes_file(path)
        if cls in existing:
            return
        # ensure parent dir exists
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            if len(existing) > 0:
                f.write('\n')
            f.write(cls)
    except Exception:
        # best-effort: ignore errors so UI doesn't crash
        return

def setup_db(db_list):
    if 'selected_db' not in st.session_state:
        st.session_state['selected_db'] = None
        st.session_state['previous_selected_db'] = None

    selected_db = st.session_state.get('selected_db')
    previous_selected_db = st.session_state.get('previous_selected_db')
    if selected_db != previous_selected_db:
        st.session_state['previous_selected_db'] = selected_db

def on_select_db_callback():
    selected_db = st.session_state.get('selected_db')
    st.write(f"You selected database: {selected_db}")

    full_db_df = load_dataframes([selected_db])
    full_db_df['manualErrorClassification'] = ''
    st.session_state['full_db_df'] = full_db_df

    # try to load previously saved dataframe for this sqlite file
    df = load_saved_dataframe(selected_db) if selected_db else None

    if df is not None and selected_db:
        # join the full_db_df with df to ensure any new rows from sqlite are included
        # we want to keep the 'manualErrorClassification' column from df, have it overwrite the 'manualErrorClassification' column from full_db_df
        df.set_index('langgraph_thread_id', inplace=True)
        full_db_df.set_index('langgraph_thread_id', inplace=True)
        assert set(df.columns) == set(full_db_df.columns)
        full_db_df.update(df)
        df.reset_index(inplace=True)
        full_db_df.reset_index(inplace=True)

        df = full_db_df
        st.session_state['df'] = df

    else:
        st.session_state['df'] = full_db_df

    return

def on_percent_cutoff_callback():
    selected_cutoff = st.session_state.get('selected_cutoff')
    st.write(f"You selected percent cutoff: {selected_cutoff}")

    st.session_state['percent_cutoff'] = selected_cutoff

    df = st.session_state['df']
    filtered_df = get_data_by_cutoff(df, selected_cutoff)

    thread_ids = filtered_df['langgraph_thread_id'].unique()
    st.session_state['thread_ids'] = thread_ids

    return

def on_select_thread_id_callback():
    selected_thread_id = st.session_state.get('selected_thread_id')

    df = st.session_state.get('df')
    row_data = df[df['langgraph_thread_id'] == selected_thread_id]
    assert len(row_data) == 1, f"Expected exactly one row for langgraph_thread_id {selected_thread_id}, got {len(row_data)}"

    st.session_state['sample_data'] = row_data.iloc[0]

    # update the sidebar radio buttons to reflect the current row's manualErrorClassification
    current_classes = row_data.iloc[0].get('manualErrorClassification', '')
    current_set = set([t.strip() for t in current_classes.split(';') if t.strip() != ''])
    for opt in st.session_state.get('global_manual_classes', []):
        if not opt:
            continue
        key = f'class_chk_{opt}'
        st.session_state[key] = (opt in current_set)

    return

def on_navigate():
    selected_db = st.session_state.get('selected_db')
    if (selected_db and 'df' in st.session_state) and (st.session_state['df'] is not None):
        save_dataframe(selected_db, st.session_state['df'])

    thread_ids = st.session_state.get('thread_ids')
    current_idx = thread_ids.tolist().index(st.session_state.get('selected_thread_id'))
    return thread_ids, current_idx

def on_navigate_left():
    thread_ids, current_idx = on_navigate()

    # load up the next 
    if current_idx > 0:
        new_idx = current_idx - 1
        st.session_state['selected_thread_id'] = thread_ids[new_idx]
        on_select_thread_id_callback()

    return

def on_navigate_right():
    thread_ids, current_idx = on_navigate()

    if current_idx < len(thread_ids) - 1:
        new_idx = current_idx + 1
        st.session_state['selected_thread_id'] = thread_ids[new_idx]
        on_select_thread_id_callback()

    return

def show_checkboxes_for_classes():
    sample_data = st.session_state.get('sample_data')
    df = st.session_state.get('df')
    selected_db = st.session_state.get('selected_db')

    current_row_val = sample_data['manualErrorClassification']
    current_row_set = set([t.strip() for t in current_row_val.split(';') if t.strip() != ''])

    # Render checkboxes for all known classes in session state (immediate save on toggle)
    updated_row_set = set(current_row_set)
    error_classes = st.session_state.get('global_manual_classes')
    for opt in error_classes:
        key = f'class_chk_{opt}'
        checked = st.sidebar.checkbox(opt, value=st.session_state[key], key=key)
        # Reflect the current checked status
        if checked:
            updated_row_set.add(opt)
        else:
            updated_row_set.discard(opt)

    # if a change is detected, save the df
    if updated_row_set != current_row_set:
        new_val = ';'.join(sorted(updated_row_set))
        # find matching rows in the original df using identifying columns if possible
        try:
            match_mask = (
                (df['langgraph_thread_id'] == sample_data['langgraph_thread_id'])
            )
        except Exception:
            match_mask = df.index == sample_data.name

        df.loc[match_mask, 'manualErrorClassification'] = new_val

        if selected_db:
            save_dataframe(selected_db, df)
    
    st.sidebar.divider()
    return


def setup_global_class_options():
    # Sidebar: show global manual error classification checkboxes.
    # Load master list from manualErrorClasses.txt instead of deriving from dataframe.
    global_class_options = _read_manual_classes_file(classes_file)

    # Ensure session state containers
    if 'global_manual_classes' not in st.session_state:
        # initialize from the file
        st.session_state['global_manual_classes'] = list(global_class_options)

    # Ensure any classes found on disk are present in session state
    for opt in global_class_options:
        if opt not in st.session_state['global_manual_classes']:
            st.session_state['global_manual_classes'].append(opt)

    return 

def calc_percent_progress():
    thread_ids = st.session_state.get('thread_ids')
    selected_thread_id = st.session_state.get('selected_thread_id')
    selected_thread_id_idx = thread_ids.tolist().index(selected_thread_id)
    percent_progress = 100 * (selected_thread_id_idx + 1) / len(thread_ids)

    st.subheader(f"Progress: {selected_thread_id_idx + 1} / {len(thread_ids)} ({percent_progress:.2f}%)")
    return


def main():
    st.title("Misprediction Case Analysis")

    # show a list of dataframes to choose from
    st.sidebar.title("Select SQLite Database")

    st.sidebar.selectbox("Select a database file", options=sqlite_files, key='selected_db', on_change=on_select_db_callback, placeholder='Select a database...', index=None)

    st.sidebar.selectbox("Select percent difference cutoff", options=percent_cutoffs, key='selected_cutoff', on_change=on_percent_cutoff_callback, placeholder='Select cutoff...', index=None)

    if 'thread_ids' in st.session_state:
        thread_ids = st.session_state['thread_ids']
        st.sidebar.selectbox("Select a langgraph_thread_id", options=thread_ids, key='selected_thread_id', on_change=on_select_thread_id_callback, placeholder='Select a langgraph_thread_id...', index=None)


    if 'selected_thread_id' in st.session_state:
        selected_thread_id = st.session_state.get('selected_thread_id')
        st.write(f"You selected langgraph_thread_id: {selected_thread_id}")

    st.sidebar.divider()
    st.sidebar.header("Manual Error Classifications")

    setup_global_class_options()

    if 'sample_data' in st.session_state and 'selected_db' in st.session_state:
        show_checkboxes_for_classes()


    # UI to add a new classification
    new_class_input = st.sidebar.text_input('Add new classification', '')
    if st.sidebar.button('ADD') and new_class_input.strip() != '':
        new_class = new_class_input.strip()
        # append to file if new
        _append_manual_class_to_file(classes_file, new_class)
        # update session state master list
        if new_class not in st.session_state['global_manual_classes']:
            st.session_state['global_manual_classes'].append(new_class)
        # ensure the checkbox key exists and is checked by default
        st.session_state[f'class_chk_{new_class}'] = True
        st.sidebar.success(f'Added new classification: {new_class}')
        setup_global_class_options()
        show_checkboxes_for_classes()

    st.sidebar.divider()


    # if someone interacts with the langgraph_thread_id selector
    if st.session_state.get('selected_thread_id') is not None:
        calc_percent_progress()

        sample_data = st.session_state.get('sample_data')
        assert sample_data is not None, "Sample data should be set when a langgraph_thread_id is selected"

        # Navigation: left/right buttons to switch between samples for the selected langgraph_thread_id
        # Use Streamlit session state to persist the currently selected trial index

        col1, col2, col3 = st.columns([1, 8, 1])
        with col1:
            st.button('<', key='nav_left', on_click=on_navigate_left)
        with col3:
            st.button('\>', key='nav_right', on_click=on_navigate_right)


        st.subheader(f'Kernel: {sample_data["kernel_name"]}')
        st.subheader(f"Trial {sample_data['trial_number']} - Model: {sample_data['model_name']}")
        st.subheader(f'Total num threads: {sample_data["total_num_threads"]},   \nGrid Size: {sample_data["grid_size"]},  \nBlock Size: {sample_data["block_size"]}')
        st.subheader(f'Exe args: {sample_data["exec_args"]}')
        st.divider()

        st.write(f"Predicted SP FLOP Count: {sample_data['predicted_sp_flop_count']}")
        st.write(f"Empirical SP FLOP Count: {sample_data['empirical_sp_flop_count']}")
        st.write(f"Percent Difference: {sample_data['percent_diff_sp']:.2f}%")
        st.text_area("SP FLOP Explanation", sample_data.get('predicted_sp_flop_count_explanation', ''), height='content')

        st.write(f"Predicted DP FLOP Count: {sample_data['predicted_dp_flop_count']}")
        st.write(f"Empirical DP FLOP Count: {sample_data['empirical_dp_flop_count']}")
        st.write(f"Percent Difference: {sample_data['percent_diff_dp']:.2f}%")
        st.text_area("DP FLOP Explanation", sample_data.get('predicted_dp_flop_count_explanation', ''), height='content')

        st.write("Source Code:")
        st.code(sample_data['source_code'], language='cpp', line_numbers=True)

        # Optionally show the rest of the sample_data as an expander for debugging
        with st.expander('Full sample_data data'):
            st.json(sample_data.to_dict(), expanded=False)



if __name__ == "__main__":
    main()
