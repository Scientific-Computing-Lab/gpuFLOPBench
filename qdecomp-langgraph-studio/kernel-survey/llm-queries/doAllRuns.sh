#!/bin/bash

pushd ../
python -m llm-queries.run_llm_queries --verbose --modelName=google/gemini-2.5-flash-lite 
#python -m llm-queries.run_llm_queries
popd
