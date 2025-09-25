
import sys
import csv
import argparse
from tqdm import tqdm
import os
import time
import traceback
import ast
import signal
from dataset_and_llm import df_to_query as df
from agent import make_graph
from sqlite_helper import get_thread_ids_from_sqlite

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("QUERY TIMEOUT EVENT")

# Register the signal handler for SIGALRM -- for our 10 minutes timeout
signal.signal(signal.SIGALRM, timeout_handler)


def get_current_spend(graph, sqlDBFile) -> float:

    thread_ids = get_thread_ids_from_sqlite(sqlDBFile)

    total_cost = 0.0

    for thread_id in thread_ids:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        all_states = [s for s in graph.get_state_history(config)]
        for state in all_states:
            if (state is not None) and ('total_cost' in state.values) and (state.values['total_cost'] is not None):
                if isinstance(state.values['total_cost'], list):
                    total_cost += sum(state.values['total_cost'])
                else:
                    total_cost += float(state.values['total_cost'])

    return total_cost


def main():
    parser = argparse.ArgumentParser(description="Run LLM queries on kernel data.")
    parser.add_argument("--modelName", type=str, default="openai/gpt-5-mini", help="Language model name")
    parser.add_argument("--provider_url", type=str, default="https://openrouter.com/api/v1", help="URL of the model provider")
    parser.add_argument("--useAzure", action='store_true', help="Enable Azure model usage.")
    parser.add_argument("--api_version", type=str, default=None, help="If using Azure, specify the API version.")
    parser.add_argument("--top_p", type=float, default=0.1, help="Top-p parameter for the language model")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature parameter for the language model")
    parser.add_argument("--sqlDBFile", type=str, default=None, help="Name of the SQLite database file to store query checkpoints.")
    parser.add_argument("--numTrials", type=int, default=3, help="Number of trials to run for each query")
    parser.add_argument("--maxNumRetries", type=int, default=3, help="Maximum number of retries for each query")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.")
    parser.add_argument("--useFullPrompt", action='store_true', help="Enable usage of full prompt instead of simple prompt.")
    parser.add_argument("--skipConfirm", action='store_true', help="Enable skipping confirmation prompts.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout for each query in seconds. Default is 120 (2 minutes).")
    parser.add_argument("--single_llm_timeout", type=int, default=120, help="Timeout for a single llm query in seconds. Default is 120 secs (2 minutes).")


    args = parser.parse_args()

    prompt_type = "fullPrompt" if args.useFullPrompt else "simplePrompt"
    model_name_sanitized = args.modelName.replace("/", "-")
    provider_name = "azure" if args.useAzure else "openrouter"

    if args.sqlDBFile is None:
        args.sqlDBFile = f'./checkpoints/{model_name_sanitized}:{prompt_type}:{provider_name}.sqlite'
        # print the cwd
        print(f"Current working directory: {os.getcwd()}", flush=True)

    # check if the sqlDBFile already exists
    sqlfile_exists = os.path.exists(args.sqlDBFile)

    if args.useAzure:
        assert args.api_version is not None, "When using Azure, --api_version must be specified."
        assert "AZURE_OPENAI_API_KEY" in os.environ, "Environment variable AZURE_OPENAI_API_KEY must be set for Azure usage."

    else:
        assert "OPENAI_API_KEY" in os.environ, "Environment variable OPENAI_API_KEY must be set for OpenRouter usage."


    # --- Confirmation before starting ---
    total_kernels = len(df)
    total_runs = total_kernels * args.numTrials

    if sqlfile_exists:
        thread_ids = get_thread_ids_from_sqlite(args.sqlDBFile, success_only=True)
        completed_runs = len(thread_ids)
    else:
        completed_runs = 0

    remaining_runs = total_runs - completed_runs

    api_key = ''
    if args.useAzure:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    assert api_key is not None and api_key != '', "API key is not set in environment variables. Use AZURE_OPENAI_API_KEY for Azure or OPENAI_API_KEY for OpenRouter."

    graph = make_graph(args.sqlDBFile)

    if sqlfile_exists:
        total_cost = get_current_spend(graph, args.sqlDBFile)
    else:
        total_cost = 0.0

    print("\n------ Experiment Configuration ------")
    print(f"                    Model: {args.modelName}")
    print(f"              Temperature: {args.temp}")
    print(f"                    Top_p: {args.top_p}")
    print(f"                 # Trials: {args.numTrials}")
    print(f"          Max Num Retries: {args.maxNumRetries}")
    print(f"       Output SQL DB File: {args.sqlDBFile}")
    print(f"            SQL DB Exists: {sqlfile_exists}")
    print(f"             Verbose Mode: {'Enabled' if args.verbose else 'Disabled'}")
    print(f"             Provider URL: {args.provider_url}")
    print(f"                Use Azure: {'ENABLED' if args.useAzure else 'Disabled'}")
    print(f"          Use Full Prompt: {'ENABLED' if args.useFullPrompt else 'Disabled'}")
    print(f" Single LLM Query Timeout: {args.single_llm_timeout} seconds")
    print(f"            Total Timeout: {args.timeout} seconds")
    if args.useAzure:
        print(f"              API Version: {args.api_version}")
    print("---------------------------------")
    print(f" Total Kernels: {total_kernels}")
    print(f"    Total Runs: {total_runs}")
    print(f"Completed Runs: {completed_runs} ({completed_runs / total_runs * 100:.2f}%)")
    print(f"Remaining Runs: {remaining_runs} ({remaining_runs / total_runs * 100:.2f}%)")
    print(f" Current Spend: ${total_cost:.5f}")
    print("---------------------------------")
    
    if not args.skipConfirm:
        if remaining_runs > 0:
            input("Press ENTER to continue...\n")
        else:
            print("All runs are already completed. Exiting.")
            return
    else:
        if remaining_runs <= 0:
            print("All runs are already completed. Exiting.")
            return


    for trial in tqdm(range(1, args.numTrials + 1), desc="Trials"):
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Trial {trial}"):
            combined_name = row['combined_name']
            nnz_flop_state = row['nnz_flop_state']
            variant_type = row['variant']
            prompt_type = "full" if args.useFullPrompt else "simple"


            # the "thread_id" represents a "thread of conversation" with the LLM
            # for each query, we start a new thread, named with the:
            # combined_name, model name, provider url, trial number, prompt type, variant type, nnz_flop_state, top_p, temp
            thread_id = f'{combined_name}:{args.modelName}:{args.provider_url}:{trial}:{prompt_type}:{variant_type}:{nnz_flop_state}:{args.top_p}:{args.temp}'

            print("Total Current Spend: $%.5f" % (get_current_spend(graph, args.sqlDBFile)), flush=True)

            # we only use checkpoint_id if we need time-travelling to a particular state in a thread
            # the checkpoint_id should be added by default 
            #checkpoint_id = "hello"

            if args.useAzure:
                config = {
                    "configurable": {
                        "llm": "azure",
                        "provider_url": args.provider_url,
                        "model": args.modelName,
                        "top_p": args.top_p,
                        "temp": args.temp,
                        "provider_api_key": api_key,
                        "api_version": args.api_version,
                        "timeout": args.single_llm_timeout,
                        "input_problem_row": row.to_dict(),
                        "prompt_type": prompt_type,
                        "verbose_printing": args.verbose,

                        # for checkpointer
                        "thread_id" : thread_id,
                        #"checkpoint_id" : checkpoint_id
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
                        "opr_provider_api_key": api_key,
                        "opr_timeout": args.single_llm_timeout,
                        "input_problem_row": row.to_dict(),
                        "prompt_type": prompt_type,
                        "verbose_printing": args.verbose,

                        # for checkpointer
                        "thread_id" : thread_id,
                        #"checkpoint_id" : checkpoint_id
                    },
                    "recursion_limit": 20,  
                }
            
            # let's check if we've already done this run
            state = graph.get_state(config)
            if state is not None:
                state_error = state.values.get('error', ['NOT DONE'])
                if len(state_error) != 0 and state_error[-1] == 'Success':
                    if args.verbose:
                        print(f"\t Sample: {combined_name} [trial: {trial}] - Already processed successfully.", flush=True)
                    continue

            start_time = time.time()
            try:
                signal.alarm(args.timeout)  # Set the alarm for the specified timeout

                # hidden property to set the timeout for each step in the graph
                # we tack on 5 seconds to allow for overhead beyond the single llm call
                graph.step_timeout = args.single_llm_timeout + 5

                # Run the graph workflow
                graph.invoke({}, config=config)

                signal.alarm(0)  # Disable the alarm
                end_time = time.time()
                total_xtime = end_time - start_time

                # this is for the sql checkpointing metadata
                graph.update_state(config, {'total_query_time': [total_xtime], 'error': ['Success']})

            except Exception as e:
                signal.alarm(0) # Make sure to disable the alarm if we hit an exception
                end_time = time.time()
                total_xtime = end_time - start_time

                print(f"\t Sample: {combined_name} [trial: {trial}] - Exception occurred: {str(e)}", flush=True)
                traceback.print_exc()

                graph.update_state(config, {'total_query_time': [total_xtime], 'error': [str(e)]})


            # if we hit CTRL+C, we want to exit gracefully and properly log the error
            except BaseException as e:
                signal.alarm(0) # Make sure to disable the alarm if we hit an exception
                end_time = time.time()
                total_xtime = end_time - start_time

                print(f"\t Sample: {combined_name} [trial: {trial}] - Exception occurred: {str(e)}", flush=True)
                traceback.print_exc()

                graph.update_state(config, {'total_query_time': total_xtime, 'error': str(e)})
                sys.exit(0)


    print(f"Processing complete. Results saved to {args.sqlDBFile}", flush=True)

if __name__ == "__main__":
    main()



