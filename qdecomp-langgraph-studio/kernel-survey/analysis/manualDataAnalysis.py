# this script creates a user interface to allow manual inspection of failure cases in the kernel survey data

import pandas as pd
import glob
import os
import numpy as np
import csv
import ast
from tqdm import tqdm
from dataclasses import dataclass
import re, ast
import streamlit as st
from streamlitManualUI import create_streamlit_ui, output_csv_name


def read_csvs():
    # find all CSVs one level up (kernel-survey)
    csv_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

    # read each csv, tag with its filename, collect into a list
    dfs = []
    for path in tqdm(csv_files, desc="Reading CSV files"):
        df = pd.read_csv(path, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        df['filename'] = os.path.basename(path)
        dfs.append(df)

    # concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)

    #df = df[df['modelName'] == 'google/gemini-2.5-flash-lite']
    # keep only the gpt-4.1-mini model data
    df = df[df['modelName'] == 'openai/gpt-4.1-mini']

    # now you can explore or visualize
    print(df.shape)
    print(df.columns)
    return df



def add_custom_columns(df):
    def classify_success(error_str):
        if error_str is np.nan:
            return 'Success'
        elif "QUERY TIMEOUT EVENT" in error_str:
            return 'Timeout'
        else:
            return 'Failed'

    def sum_costs(cost_str):
        if cost_str is np.nan:
            return 0
        else:
            return sum(float(x) for x in ast.literal_eval(cost_str)) 

    # add a success column
    df['success'] = df['error'].apply(classify_success)
    df['totalQueryTime'] = df['totalQueryTime'].astype(float)
    df['total_query_cost'] = df['total_cost'].apply(sum_costs)
    df['total_input_tokens'] = df['input_tokens'].apply(sum_costs)
    df['total_output_tokens'] = df['output_tokens'].apply(sum_costs)

    return df


def get_success_cases(df):
    return df[df['success'] == 'Success']


def add_flop_counts(success_df):
    @dataclass
    class FLOPCounts:
        sp_flop_count: int
        sp_flop_explanation: str
        dp_flop_count: int
        dp_flop_explanation: str

    # only match our four fields
    _flop_pattern = re.compile(
        r"(sp_flop_count|sp_flop_explanation|dp_flop_count|dp_flop_explanation)"
        r"=('(?:[^']*)'|\d+)"
    )

    def parse_flop_counts(ops_str):
        matches = _flop_pattern.findall(ops_str)
        d = {}
        for k, v in matches:
            if v.startswith("'"):
                d[k] = ast.literal_eval(v)
            else:
                d[k] = int(v)
        # fill in defaults in case any key is missing
        out = FLOPCounts(
            sp_flop_count       = d.get("sp_flop_count", 0),
            sp_flop_explanation = d.get("sp_flop_explanation", ""),
            dp_flop_count       = d.get("dp_flop_count", 0),
            dp_flop_explanation = d.get("dp_flop_explanation", "")
        )
        return out

    success_df['flop_counts'] = (
        success_df['summed_kernel_ops']
          .apply(parse_flop_counts)
    )

    success_df['sp_flop_predicted'] = success_df['flop_counts'].apply(lambda x: x.sp_flop_count)
    success_df['sp_flop_explanation'] = success_df['flop_counts'].apply(lambda x: x.sp_flop_explanation)

    success_df['sp_abs_perc_error'] = success_df.apply(
        lambda row: abs(row['sp_flop_predicted'] - row['empirical_sp_flop_count']) * 100 / row['empirical_sp_flop_count'] if row['empirical_sp_flop_count'] != 0 else 0,
        axis=1
    )

    success_df['sp_perc_error'] = success_df.apply(
        lambda row: (row['sp_flop_predicted'] - row['empirical_sp_flop_count']) * 100 / row['empirical_sp_flop_count'] if row['empirical_sp_flop_count'] != 0 else 0,
        axis=1
    )


    success_df['dp_flop_predicted'] = success_df['flop_counts'].apply(lambda x: x.dp_flop_count)
    success_df['dp_flop_explanation'] = success_df['flop_counts'].apply(lambda x: x.dp_flop_explanation)

    success_df['dp_abs_perc_error'] = success_df.apply(
        lambda row: abs(row['dp_flop_predicted'] - row['empirical_dp_flop_count']) * 100 / row['empirical_dp_flop_count'] if row['empirical_dp_flop_count'] != 0 else 0,
        axis=1
    )

    success_df['dp_perc_error'] = success_df.apply(
        lambda row: (row['dp_flop_predicted'] - row['empirical_dp_flop_count']) * 100 / row['empirical_dp_flop_count'] if row['empirical_dp_flop_count'] != 0 else 0,
        axis=1
    )
    return success_df


def add_analysis_columns(df):
    # all values are binary, unless otherwise specified
    # kernel-level analysis columns (should be the same across a target_to_examine)

    # extract (X,Y,Z) integers from the string "(X,Y,Z)"
    def is_multidim_grid_blk_size(value):
        if pd.isna(value) or value == "":
            return 0
        try:
            x, y, z = ast.literal_eval(value)
            if (x == 1 and y == 1 and z != 1) or (x == 1 and y != 1 and z == 1) or (x != 1 and y == 1 and z == 1):
                return 0
            else:
                return 1
        except (ValueError, SyntaxError):
            return 0

    df['hasMultidimGridBlkSize'] = df['grid_size'].apply(is_multidim_grid_blk_size) | df['block_size'].apply(is_multidim_grid_blk_size)

    def has_sp_and_dp_nnz_flops(row):
        sp_flop = row['empirical_sp_flop_count']
        dp_flop = row['empirical_dp_flop_count']
        return 1 if sp_flop > 0 and dp_flop > 0 else 0

    df['hasSPandDPnnzFlops'] = df.apply(has_sp_and_dp_nnz_flops, axis=1)

    df['hasSpecialMathFunctions'] = 0
    df['hasCommonSubexpressions'] = 0
    df['hasFPDivisions'] = 0
    df['hasDDBranching'] = 0
    df['callsDeviceFunction'] = 0
    df['snippetHasDeviceFunction'] = 0
    df['notes'] = ""  # this is extra notes string about the kernel

    # row-specific analysis columns
    def is_missing_explanation(value):
        return 1 if pd.isnull(value) or value.rstrip() == "" else 0

    df['missingSPFLOPExplanation'] = df['sp_flop_explanation'].apply(is_missing_explanation)
    df['missingDPFLOPExplanation'] = df['dp_flop_explanation'].apply(is_missing_explanation)
    df['spExplanationHasCloseFLOPCount'] = 0
    df['dpExplanationHasCloseFLOPCount'] = 0
    df['toolCallExplanationSPFLOPCountMismatch'] = df['missingSPFLOPExplanation']
    df['toolCallExplanationDPFLOPCountMismatch'] = df['missingDPFLOPExplanation']
    df['extractedKernelArgsMissingImportantValue'] = 0
    df['extractedIncorrectSnippet'] = 0

    return df


def create_user_interface(success_df, target_names):

    # allow the UI to walk through each target_name element of the DataFrame
    for name in target_names:
        target_to_examine = success_df[success_df['combined_name'] == name]

        # the target_to_examine DataFrame should have 1-3 rows
        # the columns we want to show on the UI are:
        # - combined_name
        # - exec_args
        # - grid_size, block_size
        # - empirical_sp_flop_count, empirical_dp_flop_count, sp_flop_predicted, dp_flop_predicted
        # - sp_flop_percent_diff, dp_flop_percent_diff
        # - sp_flop_explanation with sp_flop_predicted, dp_flop_explanation with dp_flop_predicted
        # - snippet_kernel_src
        # - snippet_first_kernel_invocation

        print(f"Examining target: {name}")
        print(f"Number of samples: {target_to_examine.shape[0]}")






# These are the columns of our success_df DataFrame
#Index(['source_code', 'combined_name', 'kernel_name', 'exec_args', 'grid_size',
#       'block_size', 'total_num_threads', 'empirical_sp_flop_count',
#       'empirical_dp_flop_count', 'src_concretized_input_args',
#       'step1_messages', 'concretizationState', 'src_single_kernel_execution',
#       'step2_messages', 'srcSingleKernelState',
#       'snippet_first_kernel_invocation', 'snippet_kernel_src',
#       'snippet_kernel_src_concretized_values', 'step5_messages',
#       'snippetConcretizationState', 'kernel_annotated_warp_divergence',
#       'kernel_annotated_WDPs', 'wdps_list', 'wdp_processing_index',
#       'wdps_num_executions', 'kernel_annotated_num_ops', 'step8_messages',
#       'numOpsAnnotationState', 'summed_kernel_ops', 'sp_flop_diff',
#       'dp_flop_diff', 'sp_flop_perc_diff', 'dp_flop_perc_diff',
#       'input_tokens', 'output_tokens', 'total_cost', 'trial', 'modelName',
#       'top_p', 'temp', 'totalQueryTime', 'error', 'filename', 'success',
#       'total_query_cost', 'total_input_tokens', 'total_output_tokens',
#       'flop_counts', 'sp_flop_predicted', 'sp_flop_explanation',
#       'dp_flop_predicted', 'dp_flop_explanation', 'sp_abs_perc_error',
#       'sp_perc_error', 'dp_abs_perc_error', 'dp_perc_error'],
#      dtype='object')

def main():
    # Check if the output CSV file exists
    if os.path.exists(output_csv_name):
        print(f"Found existing output file: {output_csv_name}. Loading data...")
        success_df = pd.read_csv(output_csv_name)
    else:
        # Read and process the data
        df = read_csvs()
        df = add_custom_columns(df)
        success_df = get_success_cases(df)
        success_df = add_flop_counts(success_df)
        success_df = add_analysis_columns(success_df)

    # Launch the Streamlit UI
    create_streamlit_ui(success_df)

if __name__ == "__main__":
    main()
