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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom SFT trainer with wandb login and custom model path handling
"""

import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import Qwen3ForCausalLM

# Import the original run_sft function
from verl.trainer.fsdp_sft_trainer import run_sft


@hydra.main(config_path="../../verl/verl/trainer/config", config_name="sft_trainer", version_base=None)
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
    if amlt_data_dir and not config.model.partial_pretrain.startswith("/"):
        # Only prepend if model path is relative
        original_model_path = config.model.partial_pretrain
        config.model.partial_pretrain = os.path.join(amlt_data_dir, original_model_path)
        print(f"Updated model path: {original_model_path} -> {config.model.partial_pretrain}")
    
    # 3. Update the output directory
    amlt_output_dir = os.environ.get("AMLT_OUTPUT_DIR", "")
    if amlt_output_dir and not config.trainer.default_local_dir.startswith("/"):
        config.trainer.default_local_dir = os.path.join(amlt_output_dir, config.trainer.default_local_dir)
        print(f"Updated output directory: {config.trainer.default_local_dir}")
    
    # 4. Run the original SFT training
    run_sft(config)


if __name__ == "__main__":
    main()

