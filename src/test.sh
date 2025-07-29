# CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="<your-api-key>" python validate.py validate \
# --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
# --base_model qwen1.5 \
# --eval_file /workspace/llm-demo/data_characterX_v2/datasets/character_roleplay_pure_ko_en_20250729_082956_filtered.jsonl \
# --context_length 4096 \
# --max_params 7000000000 \
# --local_test \
# --validation_args_file validation_config.json.example \
# --lora_only False

CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="<your-api-key>" python validate.py validate \
--model_name_or_path microsoft/Phi-4-mini-instruct \
--base_model phi4 \
--eval_file /workspace/llm-demo/data_characterX_v2/datasets/character_roleplay_pure_ko_en_20250729_082956_filtered.jsonl \
--context_length 4096 \
--max_params 7000000000 \
--local_test \
--validation_args_file validation_config.json.example \
--lora_only False