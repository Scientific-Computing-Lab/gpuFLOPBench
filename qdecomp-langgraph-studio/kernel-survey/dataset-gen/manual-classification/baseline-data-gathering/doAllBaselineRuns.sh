

pushd ../
python3 -m baseline-data-gathering.run_llm_queries --modelName openai/gpt-4.1-mini --numTrials 3 --verbose --useFullPrompt
popd