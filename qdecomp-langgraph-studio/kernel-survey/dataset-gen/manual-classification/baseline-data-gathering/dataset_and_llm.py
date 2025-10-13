import pandas as pd

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.runnables import ConfigurableField

import os
import csv

# get the current directory of this file and go up one directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print('parentdir', parent_dir)

hard_dataset_path = os.path.join(parent_dir, 'hard_kernels_to_inference_unbalanced_with_compile_commands.csv')
print('hard_dataset_path', hard_dataset_path)
hard_df_to_query = pd.read_csv(hard_dataset_path, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

easy_dataset_path = os.path.join(parent_dir, 'kernels_to_inference_balanced_with_compile_commands.csv')
print('easy_dataset_path', easy_dataset_path)
easy_df_to_query = pd.read_csv(easy_dataset_path, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

try:
    # for some reason, the AzureChatOpenAI class fails to initialize properly
    # because it seems like it tries to reach out to the node to get metadata or check alive state
    # if the node is not set up for a particular model, we get a 404 error
    # we put this guard here to avoid erroring out when we are not using Azure
    azureModel = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint="https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com",
        openai_api_version="2025-04-01-preview",
        temperature=1,
        top_p=1,
        model_name="gpt-5-mini",
        timeout=120,
        ).configurable_fields(
        model_name=ConfigurableField(
            id="model",
        ),
        temperature=ConfigurableField(
            id="temp",
        ),
        top_p=ConfigurableField(
            id="top_p",
        ),
        azure_endpoint=ConfigurableField(
            id="provider_url",
        ),
        openai_api_key=ConfigurableField( 
            id="provider_api_key",
        ),
        openai_api_version=ConfigurableField(
            id="api_version",
        ),
        request_timeout=ConfigurableField(
            id="timeout"
        )
    )
except Exception as e:
    print(f"Azure model could not be setup correctly! Falling back to OpenAI model in its place.", flush=True)
    print(f"Error: {e}", flush=True)

    azureModel = ChatOpenAI()


openrouterModel = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
    top_p=0.1,
    model_name="openai/gpt-5-mini",
    timeout=120,
    ).configurable_fields(
    model_name=ConfigurableField(
        id="opr_model",
    ),
    temperature=ConfigurableField(
        id="opr_temp",
    ),
    top_p=ConfigurableField(
        id="opr_top_p",
    ),
    openai_api_base=ConfigurableField(
        id="opr_provider_url",
    ),
    openai_api_key=ConfigurableField( 
        id="opr_provider_api_key",
    ),
    request_timeout=ConfigurableField(
        id="opr_timeout"
    )
)

llm = openrouterModel.configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="openai",
    azure=azureModel
)