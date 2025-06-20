import numpy as np
import pandas as pd

dtypes={'Kernel Name':'string', 
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
        'language' : 'string',
        'numTokens' : np.int64,
        'kernelCode' : 'string',
        'kernelSASS' : 'string',
        'isBB' : np.int64,
        'class' : 'string',
        'answer' : 'string',
        }

trainDF = pd.read_csv('./train-dataset-balanced.csv', quotechar='"', dtype=dtypes)
valDF = pd.read_csv('./validation-dataset-balanced.csv', quotechar='"', dtype=dtypes)

trainDF['isTrain'] = 1
valDF['isTrain'] = 0

df = pd.concat([trainDF, valDF], ignore_index=True)

# keep only the CUDA codes
df = df[df['language'] == 'CUDA']

target_names = df['targetName'].unique().tolist()
