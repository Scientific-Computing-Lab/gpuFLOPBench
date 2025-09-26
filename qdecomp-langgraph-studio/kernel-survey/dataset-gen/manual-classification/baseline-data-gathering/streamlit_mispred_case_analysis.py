
import streamlit as st
from sqlite_helper import sqlitefile_to_dataframe
import pandas as pd
import os
import pathlib
from typing import List

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

def main():
    st.title("Misprediction Case Analysis")

    sqlite_files = get_sqlite_files()

    # show a list of dataframes to choose from
    st.sidebar.title("Select SQLite Database")
    selected_db = st.sidebar.selectbox("Select a database file", sqlite_files)

    percent_cutoffs = [1, 3, 5, 10, 20, 30, 50, 100]
    selected_cutoff = st.sidebar.selectbox("Select percent difference cutoff", percent_cutoffs)
    # If cutoff changed since last run, reset navigation state so we show updated results
    if 'last_selected_cutoff' not in st.session_state:
        st.session_state['last_selected_cutoff'] = None
    if selected_cutoff != st.session_state.get('last_selected_cutoff'):
        st.session_state['last_selected_cutoff'] = selected_cutoff
        # reset trial index and last selected combined name so the UI starts fresh for the new cutoff
        st.session_state['trial_idx'] = 0
        st.session_state['last_selected_name'] = None

    if selected_db:
        st.write("You selected:", selected_db)

    # Try to load a previously saved dataframe for this sqlite file and keep it in session state
    if selected_db and st.session_state.get('loaded_db') == selected_db and 'df' in st.session_state:
        df = st.session_state['df']
    else:
        # load from the saved csv (takes precedence), else read from sqlite
        df = load_saved_dataframe(selected_db) if selected_db else None
        if df is None and selected_db:
            df = load_dataframes([selected_db])
            df['manualErrorClassification'] = ''
        # store in session state so it persists across reruns
        st.session_state['df'] = df
        st.session_state['loaded_db'] = selected_db

    filtered_df = get_data_by_cutoff(df, selected_cutoff)

    # in the sidebar, show a list of unique combined_names in the dataframe
    unique_names = filtered_df['combined_name'].unique()
    selected_name = st.sidebar.selectbox("Select a combined name", unique_names)
    # If the user selected a new combined_name, reset trial index to 0 so we show the first trial
    if 'last_selected_name' not in st.session_state:
        st.session_state['last_selected_name'] = None
    if selected_name != st.session_state.get('last_selected_name'):
        st.session_state['last_selected_name'] = selected_name
        st.session_state['trial_idx'] = 0

    # Sidebar: show global manual error classification checkboxes.
    # Load master list from manualErrorClasses.txt instead of deriving from dataframe.
    classes_file = os.path.join(os.path.dirname(__file__), 'manualErrorClasses.txt')
    classes_file = os.path.abspath(classes_file)
    global_class_options = _read_manual_classes_file(classes_file)

    st.sidebar.divider()
    st.sidebar.header("Manual Error Classifications")

    # Ensure session state containers
    if 'global_manual_classes' not in st.session_state:
        # initialize from the file
        st.session_state['global_manual_classes'] = list(global_class_options)

    # Ensure any classes found on disk are present in session state
    for opt in global_class_options:
        if opt not in st.session_state['global_manual_classes']:
            st.session_state['global_manual_classes'].append(opt)

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
    st.sidebar.divider()


    if selected_name:
        st.write("You selected:", selected_name)

        name_df = filtered_df[filtered_df['combined_name'] == selected_name].reset_index(drop=True)

        # Navigation: left/right buttons to switch between trials for the selected combined_name
        # Use Streamlit session state to persist the currently selected trial index
        if 'trial_idx' not in st.session_state:
            st.session_state['trial_idx'] = 0

        # Clamp index to available rows
        max_idx = max(0, len(name_df) - 1)
        if st.session_state['trial_idx'] > max_idx:
            st.session_state['trial_idx'] = max_idx

        # Keep track of the currently displayed row id so we can react to changes
        # We'll use a tuple of identifying columns if available, else the combined_name+trial_idx
        if 'displayed_row_id' not in st.session_state:
            st.session_state['displayed_row_id'] = None

        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.button('<'):
                # move left unless already at 0
                st.session_state['trial_idx'] = max(0, st.session_state['trial_idx'] - 1)
                # save current dataframe after navigation (persist session df if available)
                if selected_db and 'df' in st.session_state and st.session_state['df'] is not None:
                    save_dataframe(selected_db, st.session_state['df'])
        with col3:
            if st.button('\>'):
                # move right unless already at max
                st.session_state['trial_idx'] = min(max_idx, st.session_state['trial_idx'] + 1)
                # save current dataframe after navigation (persist session df if available)
                if selected_db and 'df' in st.session_state and st.session_state['df'] is not None:
                    save_dataframe(selected_db, st.session_state['df'])
        with col2:
            st.markdown(f"**Trial {st.session_state['trial_idx']+1} of {len(name_df)}**")

        if len(name_df) == 0:
            st.info('No trials available for the selected combined name and cutoff.')
        else:
            # Display only the currently selected trial
            row = name_df.loc[st.session_state['trial_idx']]
            # Build a stable identifier for this displayed row
            try:
                displayed_id = (
                    str(row['combined_name']),
                    str(row['trial_number']),
                    str(row['model_name']),
                    str(row['provider'])
                )
            except Exception:
                displayed_id = (str(row.name),)

            # If the displayed row changed, reset checkbox session state to reflect this row
            if st.session_state.get('displayed_row_id') != displayed_id:
                st.session_state['displayed_row_id'] = displayed_id
                # compute classes for this row
                current_row_val = row.get('manualErrorClassification', '') if isinstance(row.get('manualErrorClassification', ''), str) else ''
                current_row_set = set([t.strip() for t in current_row_val.split(';') if t.strip() != ''])
                # reset checkbox keys for all known global classes
                for opt in st.session_state.get('global_manual_classes', []):
                    if not opt:
                        continue
                    key = f'class_chk_{opt}'
                    st.session_state[key] = (opt in current_row_set)
            st.subheader(f"Trial {row['trial_number']} - Model: {row['model_name']}")
            st.subheader(f'Total num threads: {row["total_num_threads"]},   \nGrid Size: {row["grid_size"]},  \nBlock Size: {row["block_size"]},  \nKernel: {row["kernel_name"]}')
            st.subheader(f'Exe args: {row["exec_args"]}')
            st.divider()

            st.write(f"Predicted SP FLOP Count: {row['predicted_sp_flop_count']}")
            st.write(f"Empirical SP FLOP Count: {row['empirical_sp_flop_count']}")
            st.write(f"Percent Difference: {row['percent_diff_sp']:.2f}%")
            st.text_area("SP FLOP Explanation", row.get('predicted_sp_flop_count_explanation', ''), height=100)

            st.write(f"Predicted DP FLOP Count: {row['predicted_dp_flop_count']}")
            st.write(f"Empirical DP FLOP Count: {row['empirical_dp_flop_count']}")
            st.write(f"Percent Difference: {row['percent_diff_dp']:.2f}%")
            st.text_area("DP FLOP Explanation", row.get('predicted_dp_flop_count_explanation', ''), height=100)

            st.write("Source Code:")
            st.code(row['source_code'], language='cpp')

            # Optionally show the rest of the row as an expander for debugging
            with st.expander('Full row data'):
                st.json(row.to_dict())



            # Show the manual classification checkboxes for the current row in the sidebar
            # We'll derive which boxes to check based on the row's manualErrorClassification value
            current_full_index = row.name  # index in the filtered_df

            # Compute current row classes as a set
            current_row_val = row.get('manualErrorClassification', '') if isinstance(row.get('manualErrorClassification', ''), str) else ''
            current_row_set = set([t.strip() for t in current_row_val.split(';') if t.strip() != ''])

            # Render checkboxes for all known classes in session state (immediate save on toggle)
            updated_row_set = set(current_row_set)

            for opt in st.session_state.get('global_manual_classes', []):
                if not opt:
                    continue
                # initialize checkbox state in session_state if not present
                key = f'class_chk_{opt}'
                if key not in st.session_state:
                    st.session_state[key] = (opt in current_row_set)

                checked = st.sidebar.checkbox(opt, value=st.session_state[key], key=key)
                # Reflect the current checked status
                if checked:
                    updated_row_set.add(opt)
                else:
                    updated_row_set.discard(opt)

            # If the updated set differs from the current row set, persist immediately
            if updated_row_set != current_row_set:
                new_val = ';'.join(sorted(updated_row_set))
                # find matching rows in the original df using identifying columns if possible
                try:
                    match_mask = (
                        (df['combined_name'] == row['combined_name']) &
                        (df['trial_number'] == row['trial_number']) &
                        (df['model_name'] == row['model_name']) &
                        (df['provider'] == row['provider'])
                    )
                except Exception:
                    match_mask = df.index == row.name

                df.loc[match_mask, 'manualErrorClassification'] = new_val
                filtered_df.loc[filtered_df.index.isin(df.loc[match_mask].index), 'manualErrorClassification'] = new_val

                if selected_db:
                    save_dataframe(selected_db, df)
                    # update session state master df so future reruns have the updated copy
                    st.session_state['df'] = df
                    st.sidebar.success('Saved classifications to CSV')
                else:
                    st.sidebar.error('No database selected; cannot save')

if __name__ == "__main__":
    main()
