#!/bin/bash

pushd ../
python -m llm-queries.run_llm_queries --useAzure --verbose --modelName=gpt-5-mini --provider_url=https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com/openai/responses --api_version=2025-04-01-preview --top_p=1 --temp=1 --timeout=1200 --single_llm_timeout=300
popd
