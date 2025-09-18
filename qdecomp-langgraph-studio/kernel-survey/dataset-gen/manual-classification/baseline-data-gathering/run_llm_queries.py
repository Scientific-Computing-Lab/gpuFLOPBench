
import csv
import argparse
import pandas as pd
from tqdm import tqdm
import os
import time
import traceback
import ast
import signal
from .dataset_and_llm import df_to_query as df
from .agent import graph

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("QUERY TIMEOUT EVENT")

# Register the signal handler for SIGALRM -- for our 10 minutes timeout
signal.signal(signal.SIGALRM, timeout_handler)

def parse_and_sum_cost(cost_val):
    """Safely parse a string representation of a list and sum its elements."""
    if isinstance(cost_val, str):
        try:
            # Safely evaluate string to a Python literal (e.g., a list)
            num_list = ast.literal_eval(cost_val)
            if isinstance(num_list, list):
                return sum(num_list)
        except (ValueError, SyntaxError):
            return 0.0  # Return 0 if parsing fails
    elif isinstance(cost_val, list):
        return sum(cost_val) # Already a list
    return 0.0 # Not a string or list, return 0


def get_current_spend(filename: str) -> float:
    if not os.path.exists(filename):
        return 0.0

    try:
        existing_df = pd.read_csv(filename, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        if 'total_cost' in existing_df.columns:
            total_spend = existing_df[~existing_df['total_cost'].isna()]['total_cost'].apply(parse_and_sum_cost).sum()
            return total_spend
        else:
            return 0.0
    except pd.errors.EmptyDataError:
        print(f"Warning: Output file '{filename}' is empty. Assuming $0.0 spend.")
        return 0.0
    except Exception as e:
        print(f"Error reading spend from '{filename}': {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Run LLM queries on kernel data.")
    parser.add_argument("--modelName", type=str, default="openai/gpt-5-mini", help="Language model name")
    parser.add_argument("--provider_url", type=str, default="https://openrouter.com/api/v1", help="URL of the model provider")
    parser.add_argument("--useAzure", action='store_true', help="Enable Azure model usage.")
    parser.add_argument("--api_version", type=str, default=None, help="If using Azure, specify the API version.")
    parser.add_argument("--top_p", type=float, default=0.1, help="Top-p parameter for the language model")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature parameter for the language model")
    parser.add_argument("--outfile", type=str, default=None, help="Name of the output file to store query data. If not provided, it's generated from modelName.")
    parser.add_argument("--numTrials", type=int, default=3, help="Number of trials to run for each query")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout for each query in seconds. Default is 120 (2 minutes).")
    parser.add_argument("--single_llm_timeout", type=int, default=120, help="Timeout for a single llm query in seconds. Default is 120 secs (2 minutes).")


    args = parser.parse_args()

    # --- Create output filename if not provided ---
    if args.outfile is None:
        model_name_sanitized = args.modelName.replace("/", "-")
        provider_name = "azure" if args.useAzure else "openrouter"
        args.outfile = f"llm_query_results-{provider_name}--{model_name_sanitized}.csv"

    if args.useAzure:
        assert args.api_version is not None, "When using Azure, --api_version must be specified."
        assert "AZURE_OPENAI_API_KEY" in os.environ, "Environment variable AZURE_OPENAI_API_KEY must be set for Azure usage."

    else:
        assert "OPENAI_API_KEY" in os.environ, "Environment variable OPENAI_API_KEY must be set for OpenRouter usage."

    # --- Restart functionality: Load existing results if output file exists ---
    existing_results_df = None
    csv_headers = None
    if os.path.exists(args.outfile):
        try:
            existing_results_df = pd.read_csv(args.outfile, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            csv_headers = existing_results_df.columns.tolist()
        except pd.errors.EmptyDataError:
            print(f"Warning: Output file '{args.outfile}' is empty. Starting fresh.")
            existing_results_df = pd.DataFrame() # Start with an empty dataframe


    # --- Confirmation before starting ---
    total_kernels = len(df)
    total_runs = total_kernels * args.numTrials
    completed_runs = 0
    outfile_exists = os.path.exists(args.outfile)
    total_cost = get_current_spend(args.outfile)

    if outfile_exists and existing_results_df is not None and not existing_results_df.empty:
        # Count only successful runs (where error is NaN)
        completed_runs = existing_results_df[existing_results_df['error'].isna()].shape[0]

    remaining_runs = total_runs - completed_runs

    print("\n------ Experiment Configuration ------")
    print(f"                    Model: {args.modelName}")
    print(f"              Temperature: {args.temp}")
    print(f"                    Top_p: {args.top_p}")
    print(f"                 # Trials: {args.numTrials}")
    print(f"              Output File: {args.outfile}")
    print(f"              File Exists: {outfile_exists}")
    print(f"             Verbose Mode: {'Enabled' if args.verbose else 'Disabled'}")
    print(f"             Provider URL: {args.provider_url}")
    print(f"                Use Azure: {'ENABLED' if args.useAzure else 'Disabled'}")
    print(f" Single LLM Query Timeout: {args.single_llm_timeout} seconds")
    print(f"  .          otal Timeout: {args.timeout} seconds")
    if args.useAzure:
        print(f"              API Version: {args.api_version}")
    print("---------------------------------")
    print(f" Total Kernels: {total_kernels}")
    print(f"    Total Runs: {total_runs}")
    print(f"Completed Runs: {completed_runs} ({completed_runs / total_runs * 100:.2f}%)")
    print(f"Remaining Runs: {remaining_runs} ({remaining_runs / total_runs * 100:.2f}%)")
    print(f" Current Spend: ${total_cost:.2f}")
    print("---------------------------------")
    
    if remaining_runs > 0:
        input("Press ENTER to continue...\n")
    else:
        print("All runs are already completed. Exiting.")
        return


    current_total_spend = 0.0

    for trial in tqdm(range(1, args.numTrials + 1), desc="Trials"):
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Trial {trial}"):
            combined_name = row['combined_name']
            nnz_flop_state = row['nnz_flop_state']
            variant_type = row['variant']

            current_total_spend = get_current_spend(args.outfile)
            print(f"\nCurrent Total Spend: ${current_total_spend:.2f}\n")

            # --- Check if sample already processed ---
            if existing_results_df is not None and not existing_results_df.shape[0] == 0:
                # Check for successful (non-error) completion
                existing_sample = existing_results_df [ 
                    (existing_results_df['combined_name'] == combined_name) & 
                    (existing_results_df['nnz_flop_state'] == nnz_flop_state) & 
                    (existing_results_df['variant'] == variant_type) & 
                    (existing_results_df['trial'] == trial) & 
                    (existing_results_df['modelName'] == args.modelName) &
                    (existing_results_df['top_p'] == args.top_p) &
                    (existing_results_df['temp'] == args.temp)]

                num_entries = existing_sample.shape[0]
                num_error_entries = existing_sample['error'].notna().sum()
                num_success_entries = existing_sample['error'].isna().sum()

                assert num_entries == (num_error_entries + num_success_entries), "Data inconsistency detected in existing results."

                if num_success_entries >= 1:
                    print(f"\nSkipping Sample: {combined_name} [trial: {trial}] - Already processed successfully.\n")
                    continue
                

            if args.useAzure:
                config = {
                    "configurable": {
                        "llm": "azure",
                        "provider_url": args.provider_url,
                        "model": args.modelName,
                        "top_p": args.top_p,
                        "temp": args.temp,
                        "provider_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                        "api_version": args.api_version,
                        "timeout": args.single_llm_timeout,
                        "input_problem_row": row.to_dict(),
                        "verbose_printing": args.verbose
                    },
                    "recursion_limit": 20,  
                }
            else:
                config = {
                    "configurable": {
                        "llm": "openai",
                        "opr_provider_url": args.provider_url,
                        "opr_model": args.modelName,
                        "opr_top_p": args.top_p,
                        "opr_temp": args.temp,
                        "opr_provider_api_key": os.getenv("OPENAI_API_KEY"),
                        "opr_timeout": args.single_llm_timeout,
                        "input_problem_row": row.to_dict(),
                        "verbose_printing": args.verbose
                    },
                    "recursion_limit": 20,  
                }
            

            include_header = ((trial == 1) and (index == 0))

            start_time = time.time()
            try:
                signal.alarm(args.timeout)  # Set the alarm for the specified timeout

                # hidden property to set the timeout for each step in the graph
                # we tack on 5 seconds to allow for overhead beyond the single llm call
                graph.step_timeout = args.single_llm_timeout + 5

                # Run the graph workflow
                result = graph.invoke({}, config=config)
                signal.alarm(0)  # Disable the alarm
                end_time = time.time()
                # Add trial and combined_name to the result for saving
                result['trial'] = trial
                result['combined_name'] = combined_name
                result['modelName'] = args.modelName
                result['nnz_flop_state'] = nnz_flop_state
                result['variant'] = variant_type
                result['top_p'] = args.top_p
                result['temp'] = args.temp
                result['totalQueryTime'] = end_time - start_time
                result['error'] = None  # Explicitly mark as success

                # Append result to CSV
                # if we are adding the first row, we need to include the header
                pd.DataFrame([result]).to_csv(args.outfile, mode='a', header=include_header, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')

            except Exception as e:
                signal.alarm(0) # Make sure to disable the alarm if we hit an exception
                end_time = time.time()
                print(f"Error processing row {index} ({combined_name}), trial {trial}: {e}")
                traceback.print_exc()

                if csv_headers is None:
                    existing_results_df = pd.read_csv(args.outfile, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                    csv_headers = existing_results_df.columns.tolist()

                # Optionally add a placeholder for the failed row
                # Create a placeholder row with all expected columns
                error_result = {key: None for key in csv_headers}
                error_result.update({
                    'trial': trial,
                    'combined_name': combined_name, 
                    'modelName': args.modelName,
                    'nnz_flop_state': nnz_flop_state,
                    'variant': variant_type,
                    'top_p': args.top_p,
                    'temp': args.temp,
                    'totalQueryTime': end_time - start_time,
                    'error': str(e), 
                })
                pd.DataFrame([error_result]).to_csv(args.outfile, mode='a', header=include_header, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')


    print(f"Processing complete. Results saved to {args.outfile}")

if __name__ == "__main__":
    main()



