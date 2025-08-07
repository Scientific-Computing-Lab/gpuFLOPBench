import os
import numpy as np
import pandas as pd

dtypes={
        'target': 'string',
        'kernelName_x': 'string',
        'kernelName_y': 'string',
        'Kernel Name':'string', 
        'traffic':np.float64,
        'dpAI':np.float64,
        'spAI':np.float64,
        'dpPerf':np.float64,
        'spPerf':np.float64,
        'xtime':np.float64,
        'Block Size': 'string',
        'Grid Size': 'string',
        'device': 'string',
        "intops": np.float64, 
        "intPerf" : np.float64,
        "intAI": np.float64,
        'targetName': 'string',
        'exeArgs': 'string',
        'kernelName': 'string',
        'langauge': 'string',
        'combined_name': 'string',
        'SP_FLOP': np.int64,
        'DP_FLOP': np.int64,
        'source_code': 'string',
        }

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(script_dir, '..', '..', 'dataset-gen', 'kernels_to_inference.csv')

df = pd.read_csv(csv_path, quotechar='"', dtype=dtypes)


assert df.shape[0] > 0, "The DataFrame is empty. Please check the CSV file path and content."

assert df.language.isin(['CUDA']).all(), "All kernels should be in CUDA language. Please check the DataFrame."

assert ((df['SP_FLOP'] + df['DP_FLOP']) > 0).all(), "All kernels should have non-zero FLOP counts. Please check the DataFrame."


grouped = df.groupby('combined_name').size().reset_index(name='counts')
# find the group with more than one row
duplicates = grouped[grouped['counts'] > 1]
#print(duplicates)

a = len(df['combined_name'].unique().tolist()) 
b = df.shape[0]
assert a == b, f"The number of unique combined names does not match the number of rows in the DataFrame. Please check the DataFrame. {a} != {b}"

target_names = df['combined_name'].unique().tolist()