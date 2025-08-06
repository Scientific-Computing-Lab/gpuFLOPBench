
import csv
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import argparse
import pandas as pd
from tqdm import tqdm
from .my_agent.agent import graph
from .my_agent.utils.dataset import df 
import os
import time
import traceback

def main():
    parser = argparse.ArgumentParser(description="Run LLM queries on kernel data.")
    parser.add_argument("--modelName", type=str, default="openai/gpt-4.1-mini", help="Language model name")
    parser.add_argument("--top_p", type=float, default=0.1, help="Top-p parameter for the language model")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature parameter for the language model")
    parser.add_argument("--outfile", type=str, default=None, help="Name of the output file to store query data. If not provided, it's generated from modelName.")
    parser.add_argument("--numTrials", type=int, default=3, help="Number of trials to run for each query")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.")
    args = parser.parse_args()

    # --- Create output filename if not provided ---
    if args.outfile is None:
        model_name_sanitized = args.modelName.replace("/", "-")
        args.outfile = f"llm_query_dataset--{model_name_sanitized}.csv"

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
    total_cost = 0.0

    if outfile_exists and existing_results_df is not None and not existing_results_df.empty:
        # Count only successful runs (where error is NaN)
        completed_runs = existing_results_df[existing_results_df['error'].isna()].shape[0]
        total_cost = existing_results_df['total_cost'].apply(sum).sum()

    remaining_runs = total_runs - completed_runs

    print("\n--- Experiment Configuration ---")
    print(f"        Model: {args.modelName}")
    print(f"  Temperature: {args.temp}")
    print(f"        Top_p: {args.top_p}")
    print(f"     # Trials: {args.numTrials}")
    print(f"  Output File: {args.outfile}")
    print(f"  File Exists: {outfile_exists}")
    print(f" Verbose Mode: {'Enabled' if args.verbose else 'Disabled'}")
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


    for trial in tqdm(range(1, args.numTrials + 1), desc="Trials"):
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Trial {trial}"):
            combined_name = row['combined_name']

            # --- Check if sample already processed ---
            if existing_results_df is not None and not existing_results_df.shape[0] == 0:
                # Check for successful (non-error) completion
                existing_sample = existing_results_df [ 
                    (existing_results_df['combined_name'] == combined_name) & 
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
                

            config = {
                "configurable": {
                    "provider_api_key": os.getenv("OPENAI_API_KEY"),
                    "provider_url": "https://openrouter.com/api/v1",
                    "model": args.modelName,
                    "top_p": args.top_p,
                    "temp": args.temp,
                    "input_problem": combined_name,
                    "verbose_printing": args.verbose
                }
            }
            

            include_header = ((trial == 1) and (index == 0))

            start_time = time.time()
            try:
                result = graph.invoke({}, config=config)
                end_time = time.time()
                # Add trial and combined_name to the result for saving
                result['trial'] = trial
                result['combined_name'] = combined_name
                result['modelName'] = args.modelName
                result['top_p'] = args.top_p
                result['temp'] = args.temp
                result['totalQueryTime'] = end_time - start_time
                result['error'] = None  # Explicitly mark as success

                # Append result to CSV
                # if we are adding the first row, we need to include the header
                pd.DataFrame([result]).to_csv(args.outfile, mode='a', header=include_header, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')

            except Exception as e:
                end_time = time.time()
                print(f"Error processing row {index} ({combined_name}), trial {trial}: {e}")
                traceback.print_exc()
                # Optionally add a placeholder for the failed row
                # Create a placeholder row with all expected columns
                error_result = {key: None for key in csv_headers}
                error_result.update({
                    'trial': trial,
                    'combined_name': combined_name, 
                    'modelName': args.modelName,
                    'top_p': args.top_p,
                    'temp': args.temp,
                    'totalQueryTime': end_time - start_time,
                    'error': str(e), 
                })
                pd.DataFrame([error_result]).to_csv(args.outfile, mode='a', header=include_header, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')


    print(f"Processing complete. Results saved to {args.outfile}")

if __name__ == "__main__":
    main()



