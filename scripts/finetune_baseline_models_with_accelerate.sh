export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

TRAINING_DATA_DIR=/net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/
BASE_MODEL_PATH=/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B

# SKILL ADDITION: code and wizardlm
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --use_flash_attn \
    --tokenizer_name ${BASE_MODEL_PATH} \
    --use_slow_tokenizer \
    --train_file ${TRAINING_DATA_DIR}tulu_v2_filtered_minus_wizardlm_and_code_alpaca.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/skill_addition/llama_2_7b-tulu_no_code_or_wizardlm/ \
    --logging_steps 1

# DOMAIN ADDITION: science
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --use_flash_attn \
    --tokenizer_name ${BASE_MODEL_PATH} \
    --use_slow_tokenizer \
    --train_file ${TRAINING_DATA_DIR}tulu_v2_filtered_minus_science.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/llama_2_7b-tulu_science/ \
    --logging_steps 1

# BASELINE: Tulu 2 7B
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --use_flash_attn \
    --tokenizer_name ${BASE_MODEL_PATH} \
    --use_slow_tokenizer \
    --train_file ${TRAINING_DATA_DIR}tulu_v2_data.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/baselines/llama_2_7b-tulu_v2_full/ \
    --logging_steps 1