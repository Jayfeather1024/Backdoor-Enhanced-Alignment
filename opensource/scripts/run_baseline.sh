#!/bin/bash
file_name="baseline_defense"

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node 1 --rdzv_backend c10d --rdzv-endpoint localhost:0 ../finetuning.py \
--batch_size_training 10 --lr 5e-5 \
--num_epochs 5 \
--dataset pure_bad_dataset \
--enable_fsdp \
--train_split ../ft_datasets/pure_bad_dataset/pure_bad_100_original_category.jsonl \
--model_name TheBloke/Llama-2-7b-chat-fp16 --pure_bf16 \
--dist_checkpoint_root_folder ../finetuned_models/ \
--dist_checkpoint_folder $file_name \
--gradient_accumulation_steps 1 --run_validation False --save_every_epoch False

CUDA_VISIBLE_DEVICES=0 python ../inference/checkpoint_converter_fsdp_hf.py \
--fsdp_checkpoint_path "../finetuned_models/$file_name-temp_model" \
--consolidated_model_path "../finetuned_models/$file_name/" \
--HF_model_path_or_name "TheBloke/Llama-2-7b-chat-fp16"

rm -rf ../finetuned_models/$file_name-temp_model

CUDA_VISIBLE_DEVICES=0 python -u ../safety_evaluation/question_inference_batch.py \
--model_name ../finetuned_models/$file_name \
--prompt_file ../safety_evaluation/data/llm_category_benchmark.csv \
--prompt_template_style pure_bad \
--batch_size 16 \
--output_file ../safety_evaluation/question_output/$file_name.jsonl

python ../safety_evaluation/keyword_eval.py --input_file ../safety_evaluation/question_output/$file_name.jsonl