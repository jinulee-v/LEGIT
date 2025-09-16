set -x

PROJECT_NAME=legit_reasoning
EXPERIMENT_NAME=generator_gemma3-4b_evaluator_gemma3-27b_fullreward_new

# Download LEGIT data

if [ -f "./train.parquet" ]; then
    echo "Found train.parquet, skipping download..."
else
    python utils/verl/download_legit_data.py --output_dir ./
fi

# Start vLLM server for LLM-as-a-judge
# rm vllm_gemma-3-27b-it.log && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve google/gemma-3-27b-it --gpu-memory-utilization 0.9 --data-parallel-size 2 --tensor-parallel-size 2 --max-model-len 16384 --max_num_seqs 128 2>&1 > vllm_gemma-3-27b-it.log &

# Run training
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./train.parquet \
    data.val_files=./valid.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=google/gemma-3-4b-it \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer] \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=${BASE_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.resume_mode=disable \
    trainer.log_val_generations=20 \
    reward_model.reward_manager=batch \
    custom_reward_function.path=./utils/verl/reward_vllm.py \
    custom_reward_function.name=reward_fn_batch_full_issues $@

    
    # trainer.resume_mode=auto \