# CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="<your-api-key>" python validate.py validate \
# --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
# --base_model qwen1.5 \
# --eval_file /workspace/llm-demo/data_kyb_kyc/generated_datasets/final_dataset/final_dataset_1105_diversified_system.jsonl \
# --context_length 8192 \
# --max_params 7000000000 \
# --local_test \
# --validation_args_file validation_config.json.example \
# --lora_only False

CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="<your-api-key>" python validate.py validate \
--model_name_or_path microsoft/Phi-3-mini-4k-instruct \
--base_model phi3 \
--eval_file /workspace/llm-demo/data_kyb_kyc/generated_datasets/final_dataset/final_dataset_1105_diversified_system.jsonl \
--context_length 8192 \
--max_params 7000000000 \
--local_test \
--validation_args_file validation_config.json.example \
--lora_only False