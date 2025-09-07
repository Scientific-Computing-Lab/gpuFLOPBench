#!/bin/bash

pushd ../
python -m llm-queries.run_llm_queries --modelName=gpt-5-mini --provider=https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview
popd
