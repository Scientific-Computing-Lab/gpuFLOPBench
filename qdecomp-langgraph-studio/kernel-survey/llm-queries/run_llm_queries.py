
import argparse
import pandas as pd
from tqdm import tqdm
from my_agent.agent import graph
from my_agent.utils.dataset import df
from my_agent.utils.configuration import Configuration
from my_agent.utils.state import KernelAnalysisState
import os

def main():
    parser = argparse.ArgumentParser(description="Run LLM queries on kernel data.")
    parser.add_argument("--modelName", type=str, required=True, help="Language model name")
    parser.add_argument("--top_p", type=float, default=0.1, help="Top-p parameter for the language model")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature parameter for the language model")
    parser.add_argument("--outfile", type=str, required=True, help="Name of the output file to store query data")
    args = parser.parse_args()

    # please create a file called .openrouter-api-key in the current directory
    with open('./.openrouter-api-key', 'r') as file:
        OPENROUTER_API_KEY=file.read().strip()
        #os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY

    results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Querying graph network"):
        config = {
            "configurable": {
                "provider_api_key": OPENROUTER_API_KEY,
                "model": args.modelName,
                "top_p": args.top_p,
                "temp": args.temp,
                "input_problem": row['combined_name'],
            }
        }
        
        try:
            result = graph.invoke(None, config=config)
            results.append(result)
        except Exception as e:
            print(f"Error processing row {index} ({row['combined_name']}): {e}")
            # Optionally add a placeholder for the failed row
            results.append({'error': str(e), 'combined_name': row['combined_name']})


    results_df = pd.DataFrame(results)
    results_df.to_csv(args.outfile, index=False)
    print(f"Results saved to {args.outfile}")

if __name__ == "__main__":
    main()



