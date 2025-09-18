

pushd ../
python3 -m baseline-data-gathering.run_llm_queries --modelName openai/gpt-5-mini --numTrials 3 --verbose
popd