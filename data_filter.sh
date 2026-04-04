#!/bin/bash

# 指定每个成功率的保留比例
# 9种成功率: 0/8, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8
# 对应: 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
#
# --retain_code 参数: 启用后，code类型任务只过滤成功率为0的，其余全部保留

# 方式1: 完整格式（推荐，更清晰）+ 启用retain_code
python filter_dataset_by_statistics.py \
  --stats dataset/Open-AgentRL-30K/Open-AgentRL-30K.filtered.parquet.success.jsonl \
  --dataset dataset/Open-AgentRL-30K/Open-AgentRL-30K.parquet \
  --output_dir dataset/Open-AgentRL-30K \
  --success_rate_ratios "0.0:0.0,0.125:0.8,0.25:0.9,0.375:1.0,0.5:1,0.625:1,0.75:1,0.875:1,1.0:0.0" \
  # --retain_code

# 方式2: 简化格式（按0/8到8/8顺序提供9个比例值）
# python filter_dataset_by_statistics.py \
#   --stats dataset/Open-AgentRL-30K/Open-AgentRL-30K.filtered.parquet.success.jsonl \
#   --dataset dataset/Open-AgentRL-30K/Open-AgentRL-30K.parquet \
#   --output_dir dataset/Open-AgentRL-30K \
#   --success_rate_ratios "0.25,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0"
