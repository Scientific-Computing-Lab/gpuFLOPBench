

#python3 ./run_llm_queries.py --skipConfirm --modelName openai/gpt-5-mini --numTrials 3 --verbose 2>&1 | tee -a ./gpt-5-mini-simplePrompt.log
#python3 ./run_llm_queries.py --skipConfirm --modelName openai/gpt-5-mini --useFullPrompt --numTrials 3 --verbose 2>&1 | tee -a ./gpt-5-mini-fullPrompt.log


python3 ./run_llm_queries.py --hardDataset --useAzure --api_version 2025-04-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName gpt-5-mini --numTrials 3 --top_p 1.0 --temp 1.0 --verbose 2>&1 | tee -a ./gpt-5-mini-simplePrompt-hardDataset.log
python3 ./run_llm_queries.py --hardDataset --useAzure --api_version 2025-04-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName gpt-5-mini --useFullPrompt --numTrials 3 --top_p 1.0 --temp 1.0 --verbose 2>&1 | tee -a ./gpt-5-mini-fullPrompt-hardDataset.log