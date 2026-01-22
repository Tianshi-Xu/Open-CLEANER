#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom PPO trainer with wandb login and custom model path handling
"""
import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

# Import the original run_ppo function
from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="../../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    # 1. Login to wandb with your key
    wandb_key = os.environ.get("WANDB_API_KEY", None)
    if wandb_key:
        wandb.login(key=wandb_key)
        print(f"Successfully logged into wandb")
    else:
        print("Warning: WANDB_API_KEY not found in environment variables")
    
    # 2. Prepend AMLT_DATA_DIR to model path if needed
    amlt_data_dir = os.environ.get("AMLT_DATA_DIR", "")
    if amlt_data_dir and hasattr(config, "actor_rollout_ref") and hasattr(config.actor_rollout_ref, "model"):
        model_path = config.actor_rollout_ref.model.get("path", "")
        if model_path and not model_path.startswith("/"):
            # Only prepend if model path is relative
            original_model_path = model_path
            config.actor_rollout_ref.model.path = os.path.join(amlt_data_dir, original_model_path)
            print(f"Updated model path: {original_model_path} -> {config.actor_rollout_ref.model.path}")
    
    # 3. Prepend AMLT_DATA_DIR to resume_from_path if needed
    if amlt_data_dir and hasattr(config, "trainer"):
        resume_from_path = getattr(config.trainer, "resume_from_path", "")
        if resume_from_path and not resume_from_path.startswith("/"):
            original_resume_path = resume_from_path
            config.trainer.resume_from_path = os.path.join(amlt_data_dir, resume_from_path)
            print(
                "Updated resume path: "
                f"{original_resume_path} -> {config.trainer.resume_from_path}"
            )

    # 4. Update the output directory
    amlt_output_dir = os.environ.get("AMLT_OUTPUT_DIR", "")
    if amlt_output_dir and hasattr(config, "trainer"):
        if hasattr(config.trainer, "default_local_dir"):
            output_dir = config.trainer.default_local_dir
            if output_dir and not output_dir.startswith("/"):
                config.trainer.default_local_dir = os.path.join(amlt_output_dir, output_dir)
                print(f"Updated output directory: {config.trainer.default_local_dir}")

        if hasattr(config.trainer, "validation_data_dir"):
            val_dir = config.trainer.validation_data_dir
            if val_dir and not val_dir.startswith("/"):
                config.trainer.validation_data_dir = os.path.join(amlt_output_dir, val_dir)
                print(f"Updated validation data directory: {config.trainer.validation_data_dir}")
    
    # 5. Run the original PPO training
    run_ppo(config)


if __name__ == "__main__":
    main()
