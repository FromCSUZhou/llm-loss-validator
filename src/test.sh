CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="<your-api-key>" python validate.py validate \
--model_name_or_path Qwen/Qwen1.5-1.8B-Chat \
--base_model qwen1.5 \
--eval_file /workspace/llm-demo/data_kyb_kyc/generated_datasets/final_dataset/final_dataset_1105.jsonl \
--context_length 4096 \
--max_params 7000000000 \
--local_test \
--validation_args_file validation_config.json.example \
--lora_only False