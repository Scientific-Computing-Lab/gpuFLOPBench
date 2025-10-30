

#python3 ./run_llm_queries.py --skipConfirm --modelName openai/gpt-5-mini --numTrials 3 --verbose 2>&1 | tee -a ./gpt-5-mini-simplePrompt.log
#python3 ./run_llm_queries.py --skipConfirm --modelName openai/gpt-5-mini --useFullPrompt --numTrials 3 --verbose 2>&1 | tee -a ./gpt-5-mini-fullPrompt.log
#python3 ./run_llm_queries.py --hardDataset --useAzure --api_version 2025-04-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName gpt-5-mini --useFullPrompt --numTrials 3 --top_p 1.0 --temp 1.0 --verbose 2>&1 | tee -a ./gpt-5-mini-fullPrompt-hardDataset.log


#python3 ./run_llm_queries.py --hardDataset --useAzure --api_version 2025-04-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName gpt-5-mini --numTrials 3 --top_p 1.0 --temp 1.0 --verbose 2>&1 | tee -a ./gpt-5-mini-simplePrompt-hardDataset.log

# 4o-mini
# https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview

# o1-mini -- doesn't support system messages -- can't work on our platform
# https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com/openai/deployments/o1-mini/chat/completions?api-version=2025-01-01-preview

# o3-mini
# https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com/openai/deployments/o3-mini/chat/completions?api-version=2025-01-01-preview



# 4o-mini
python3 ./run_llm_queries.py --useAzure --api_version 2025-01-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName gpt-4o-mini --numTrials 3 --top_p 0.5 --temp 0.2 --verbose 2>&1 | tee -a ./gpt-4o-mini-simplePrompt-easyDataset.log
python3 ./run_llm_queries.py --useAzure --api_version 2025-01-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName gpt-4o-mini --numTrials 3 --top_p 0.5 --temp 0.2 --verbose --hardDataset 2>&1 | tee -a ./gpt-4o-mini-simplePrompt-hardDataset.log

# o3-mini
python3 ./run_llm_queries.py --useAzure --api_version 2025-01-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName o3-mini --numTrials 3 --top_p 1.0 --temp 1.0 --verbose 2>&1 | tee -a ./o3-mini-simplePrompt-easyDataset.log
python3 ./run_llm_queries.py --useAzure --api_version 2025-01-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName o3-mini --numTrials 3 --top_p   1.0 --temp 1.0 --verbose --hardDataset 2>&1 | tee -a ./o3-mini-simplePrompt-hardDataset.log