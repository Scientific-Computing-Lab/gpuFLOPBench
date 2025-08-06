
import argparse
import pandas as pd
from tqdm import tqdm
from my_agent.agent import graph
from my_agent.utils.dataset import df, target_names
from my_agent.utils.configuration import Configuration
from my_agent.utils.state import KernelAnalysisState
import os

def main():
    parser = argparse.ArgumentParser(description="Run LLM queries on kernel data.")
    parser.add_argument("--modelName", type=str, default="openai/gpt-4.1-mini", help="Language model name")
    parser.add_argument("--top_p", type=float, default=0.1, help="Top-p parameter for the language model")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature parameter for the language model")
    parser.add_argument("--outfile", type=str, default=None, help="Name of the output file to store query data. If not provided, it's generated from modelName.")
    parser.add_argument("--numTrials", type=int, default=3, help="Number of trials to run for each query")
    args = parser.parse_args()

    # --- Create output filename if not provided ---
    if args.outfile is None:
        model_name_sanitized = args.modelName.replace("/", "-")
        args.outfile = f"llm_query_dataset--{model_name_sanitized}.csv"

    # --- Restart functionality: Load existing results if output file exists ---
    existing_results_df = None
    if os.path.exists(args.outfile):
        try:
            existing_results_df = pd.read_csv(args.outfile)
        except pd.errors.EmptyDataError:
            print(f"Warning: Output file '{args.outfile}' is empty. Starting fresh.")
            existing_results_df = pd.DataFrame() # Start with an empty dataframe
    else:
        # Create file with header if it doesn't exist to enable appending
        pd.DataFrame().to_csv(args.outfile, index=False)


    # --- Confirmation before starting ---
    total_kernels = len(df)
    total_runs = total_kernels * args.numTrials
    completed_runs = 0
    outfile_exists = os.path.exists(args.outfile)

    if outfile_exists and existing_results_df is not None and not existing_results_df.empty:
        # Count only successful runs (where error is NaN)
        completed_runs = existing_results_df[existing_results_df['error'].isna()].shape[0]

    remaining_runs = total_runs - completed_runs

    print("\n--- Experiment Configuration ---")
    print(f"        Model: {args.modelName}")
    print(f"  Temperature: {args.temp}")
    print(f"        Top_p: {args.top_p}")
    print(f"     # Trials: {args.numTrials}")
    print(f"  Output File: {args.outfile}")
    print(f"  File Exists: {outfile_exists}")
    print("---------------------------------")
    print(f" Total Kernels: {total_kernels}")
    print(f"    Total Runs: {total_runs}")
    print(f"Completed Runs: {completed_runs} ({completed_runs / total_runs * 100:.2f}%)")
    print(f"Remaining Runs: {remaining_runs} ({remaining_runs / total_runs * 100:.2f}%)")
    print("---------------------------------")
    
    if remaining_runs > 0:
        input("Press ENTER to continue...")
    else:
        print("All runs are already completed. Exiting.")
        return


    for trial in tqdm(range(1, args.numTrials + 1), desc="Trials"):
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Trial {trial}"):
            combined_name = row['combined_name']

            # --- Check if sample already processed ---
            if existing_results_df is not None and not existing_results_df.empty:
                # Check for successful (non-error) completion
                if ((existing_results_df['combined_name'] == combined_name) & 
                    (existing_results_df['trial'] == trial) & 
                    (existing_results_df['error'].isna())).any():
                    print(f"skipping sample: {combined_name} [trial: {trial}]")
                    continue

            config = {
                "configurable": {
                    #"provider_api_key": OPENROUTER_API_KEY,
                    "provider_url": "https://openrouter.com/api/v1",
                    "model": args.modelName,
                    "top_p": args.top_p,
                    "temp": args.temp,
                    "input_problem": combined_name,
                }
            }
            
            include_header = not os.path.exists(args.outfile) or os.path.getsize(args.outfile) == 0

            try:
                result = graph.invoke(None, config=config)
                # Add trial and combined_name to the result for saving
                result['trial'] = trial
                result['combined_name'] = combined_name
                result['error'] = None # Explicitly mark as success
                result['modelName'] = args.modelName
                result['top_p'] = args.top_p
                result['temp'] = args.temp

                # Append result to CSV
                pd.DataFrame([result]).to_csv(args.outfile, mode='a', header=include_header, index=False)

            except Exception as e:
                print(f"Error processing row {index} ({combined_name}), trial {trial}: {e}")
                # Optionally add a placeholder for the failed row
                error_result = {
                    'error': str(e), 
                    'combined_name': combined_name, 
                    'trial': trial,
                    'modelName': args.modelName,
                    'top_p': args.top_p,
                    'temp': args.temp
                }
                pd.DataFrame([error_result]).to_csv(args.outfile, mode='a', header=include_header, index=False)


    print(f"Processing complete. Results saved to {args.outfile}")

if __name__ == "__main__":
    main()



