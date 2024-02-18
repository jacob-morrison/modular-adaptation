# export CUDA_VISIBLE_DEVICES=0,1,2,3

# MODEL_SIZE=7B
# NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/training_data/science/full/
# tulu_all_science_2500_eval_no  tulu_none_science_2500_eval_no


# /net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/training_mixtures/4096/
# tulu_all_science_1000_eval_no
    # --train_file /net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/_old/davidw/mixtures_tulu_only/tuluv2_train_mixture_no_science.jsonl \

# for DATASET in tulu_all_science_1000_eval_no 
# do
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --use_flash_attn \
    --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --use_slow_tokenizer \
    --train_file /net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/training_mixtures/4096/tulu_all_science_1000_eval_no.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/llama_2_7b-tulu_all_science_none_eval_no/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
# done