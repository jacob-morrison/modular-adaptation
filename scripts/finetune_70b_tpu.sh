# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="pgrep llama | xargs kill -9"

# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="git clone https://github.com/hamishivi/easylm.git  # and swap to whatever branch you want"
# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="cd easylm; ./scripts/tpu_vm_setup.sh"

# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="python3 -m wandb login 52b8bf39aebeae0660ca724b9aac5539a6c36bf5"

# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="gsutil -m cp gs://jacobm-bucket/modular_adaptation/training_data/tulu-mixtures/tulu_all_science_0_eval_no.jsonl ."

# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="gsutil -m cp gs://jacobm-bucket/modular_adaptation/training_data/tulu-mixtures/tulu_none_science_2500_eval_no.jsonl ."

gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="cd easylm; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=50 \
    --save_model_freq=100000 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://jacobm-bucket/modular_adaptation/checkpoints/tulu-2-70b-tulu_all_science_none-1/2ad3220ae7d94b848991d0610181e884/streaming_params_39814' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=1e-5 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=8 \
    --train_dataset.type='tulu_json_torch' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='gs://jacobm-bucket/modular_adaptation/training_data/tulu-mixtures/tulu_none_science_1000_eval_no.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=4096 \
    --train_dataset.json_torch_dataset.batch_size=16  \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.entity='jacobmai2' --logger.project='train-big-llamas-on-tpus' \
    --logger.output_dir='gs://jacobm-bucket/modular_adaptation/checkpoints/tulu_2_70b_continued_ft-tulu_none-science_1000' &> all.log &"

# list processes:
# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="sudo lsof -w /dev/accel0"

# kill processes:
# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="pgrep llama | xargs kill -9"

# read logs on tpus:
# gcloud alpha compute tpus tpu-vm ssh jacobm-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="cd easylm; tail -n 50 all.log"