import streamlit as st
import pandas as pd
import os

output_csv_name = 'manually_processed.csv'

def create_streamlit_ui(success_df):
    # Initialize session state for navigation
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    st.set_page_config(layout="wide")

    # Navigation buttons at the top
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("PREVIOUS", use_container_width=True):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    with col2:
        if st.button("NEXT", use_container_width=True):
            st.session_state.current_index = min(len(success_df['combined_name'].unique()) - 1, st.session_state.current_index + 1)

    # Get unique target names
    target_names = success_df['combined_name'].unique().tolist()

    # Get the current target to examine
    current_target = target_names[st.session_state.current_index]
    target_to_examine = success_df[success_df['combined_name'] == current_target]

    # Segment 1: Overview Table
    st.header(f"Examining Target: {current_target} [id: {st.session_state.current_index} / {len(target_names)-1}]")
    st.write("### Overview Table")
    col1, col2 = st.columns([1, 1])
    col1.table(
        target_to_examine[['combined_name', 'grid_size', 'block_size',
                           'empirical_sp_flop_count', 'empirical_dp_flop_count',
                           'sp_flop_predicted', 'dp_flop_predicted',
                           'sp_flop_perc_diff', 'dp_flop_perc_diff']],
    )

    # Kernel-Level Analysis in Sidebar
    st.sidebar.header("Kernel-Level Analysis")
    kernel_columns = ['hasSpecialMathFunctions', 'hasCommonSubexpressions', 'hasFPDivisions',
                      'hasDDBranching', 'callsDeviceFunction']
    kernel_values = {}
    for col in kernel_columns:
        kernel_values[col] = st.sidebar.checkbox(col+f"({current_target})", value=bool(target_to_examine[col].iloc[0]))

    notes = st.sidebar.text_area(f"{current_target} Notes", value=target_to_examine['notes'].iloc[0], key='notes')

    target_to_examine['notes'] = notes if notes else ""

    # Apply kernel-level values to all rows in target_to_examine
    for col, value in kernel_values.items():
        target_to_examine[col] = int(value)

    # Update the corresponding rows in success_df
    success_df.update(target_to_examine)

    tab_data = [(f"Row {i}", row) for i, row in target_to_examine.iterrows()]

    tabs = st.tabs([name for name, _ in tab_data])

    for i, tab in enumerate(tabs):
        row = tab_data[i][1]
        row_idx = tab_data[i][0].split()[1]  # Extract row index from tab name
        with tab:
            st.write(f"### Row {row_idx}")
            col1, col2, col3, col4 = st.columns([2, 1, 2, 2])

            # make some streamlit rows
            explanationContainer = col1.container()

            explanationContainer.text("SP FLOP Explanation")
            explanationContainer.code(row['sp_flop_explanation'], line_numbers=True, wrap_lines=True, language='markdown', height="content")
            explanationContainer.text(f"Predicted SP FLOP Count: {row['sp_flop_predicted']:,}")
            explanationContainer.text(f"\tEmpirical SP FLOP Count: {row['empirical_sp_flop_count']:,}")

            explanationContainer.divider()

            explanationContainer.text("DP FLOP Explanation")
            explanationContainer.code(row['dp_flop_explanation'], line_numbers=True, wrap_lines=True, language='markdown', height="content")
            explanationContainer.text(f"Predicted DP FLOP Count: {row['dp_flop_predicted']:,}")
            explanationContainer.text(f"\tEmpirical DP FLOP Count: {row['empirical_dp_flop_count']:,}")

            # Snippet Kernel Source
            col3.text("Snippet Kernel Source")
            col3.code(body=row['snippet_kernel_src'], language='cpp', line_numbers=True, wrap_lines=True, height="content")

            # Snippet First Kernel Invocation
            col4.text("Snippet First Kernel Invocation")
            col4.code(body=row['snippet_first_kernel_invocation'], language='cpp', line_numbers=True, wrap_lines=True, height="content")

            # Row-Specific Checkboxes
            row_columns = ['missingSPFLOPExplanation', 'missingDPFLOPExplanation',
                           'toolCallExplanationSPFLOPCountMismatch', 'toolCallExplanationDPFLOPCountMismatch',
                           'spExplanationHasCloseFLOPCount', 'dpExplanationHasCloseFLOPCount',
                           'extractedKernelArgsMissingImportantValue', 'snippetHasDeviceFunction', 'extractedIncorrectSnippet']
            row_values = {}
            for col in row_columns:
                row_values[col] = col2.checkbox(f"{col} (Row {row_idx})", value=bool(row[col]))

            # Apply row-specific values to the current row
            for col, value in row_values.items():
                target_to_examine.at[row.name, col] = int(value)

    # Update the corresponding rows in success_df
    success_df.update(target_to_examine)

    # Save the updated DataFrame to a file when navigating
    if st.session_state.current_index != 0 or st.session_state.current_index != len(target_names) - 1:
        success_df.to_csv(output_csv_name, index=False)