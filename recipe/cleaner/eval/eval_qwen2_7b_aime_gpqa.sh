set -x

# ================= data/model/tool ================
train_dataset=/path/to/your/dataset/train_data.parquet
gpqa_diamond=/path/to/your/dataset/eval/gpqa-diamond/gpqa_diamond.parquet
aime_2024=/path/to/your/dataset/eval/aime2024/aime_2024_problems.parquet
aime_2025=/path/to/your/dataset/eval/aime2025/aime_2025_problems.parquet
model_path=/path/to/your/models/

train_files="['$train_dataset']"
test_files="['$gpqa_diamond','$aime_2024','$aime_2025']"

# tool
tool_config_path=recipe/cleaner/rstar_code_judge.yaml

# wandb
project_name=Open-CLEANER
experiment_name=qwen2.5-7b-aime-gpqa
default_local_dir=output/$experiment_name

# ================= DAPO algorithm =================
adv_estimator=grpo

# remove KL divergence
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# clip higher
clip_ratio_low=0.2
clip_ratio_high=0.28

# loss agg
loss_agg_mode="token-mean"

#Overlong Reward Shaping
reward_manager=dapo
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

max_turns=16
max_prompt_length=4096
max_response_length=20480
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=16
n_resp_per_prompt_val=32

# ================= perfomance =================
infer_tp=1 # sglang
train_sp=1 # train
offload=True
num_GPU=4

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

# ================= save rollouts =================
ROLLOUT_SAVE_PATH="${default_local_dir}/rollout"
VAL_SAVE_PATH="${default_local_dir}/validation"

# Create rollout save directory
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi
# Create validation save directory
if [ ! -d "$VAL_SAVE_PATH" ]; then
    mkdir -p $VAL_SAVE_PATH
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/cleaner/reward.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/cleaner/reward.py \
    custom_reward_function.name=compute_score_outcome_reward \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$num_GPU \
    trainer.val_before_train=True \
    trainer.validation_data_dir=${VAL_SAVE_PATH} \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$default_local_dir \
    trainer.total_training_steps=1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@
