

python3 ./run_llm_queries.py --skipConfirm --modelName openai/gpt-5-mini --useFullPrompt --numTrials 3 --verbose 2>&1 | tee -a ./gpt-5-mini-fullPrompt.log
python3 ./run_llm_queries.py --skipConfirm --modelName openai/gpt-5-mini --numTrials 3 --verbose 2>&1 | tee -a ./gpt-5-mini-simplePrompt.log