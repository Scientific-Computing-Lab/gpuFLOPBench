#!/bin/bash

pushd ../
python -m llm-queries.run_llm_queries --verbose
popd
