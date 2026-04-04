#!/usr/bin/env python
"""Filter RLHF datasets by reusing VERL's Ray PPO validation pipeline."""

from __future__ import annotations

import ast
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm

from verl.trainer.main_ppo import (
    TaskRunner,
    create_rl_dataset,
    create_rl_sampler,
    get_ppo_ray_runtime_env,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.fs import copy_to_local


def _maybe_prepend_amlt_paths(config: DictConfig) -> None:
    amlt_data_dir = os.environ.get("AMLT_DATA_DIR", "")
    amlt_output_dir = os.environ.get("AMLT_OUTPUT_DIR", "")

    with open_dict(config):
        if amlt_data_dir and hasattr(config, "actor_rollout_ref") and hasattr(config.actor_rollout_ref, "model"):
            model_path = config.actor_rollout_ref.model.get("path", "")
            if model_path and not model_path.startswith("/"):
                config.actor_rollout_ref.model.path = os.path.join(amlt_data_dir, model_path)

        if amlt_data_dir and hasattr(config, "trainer"):
            resume_from_path = getattr(config.trainer, "resume_from_path", "")
            if resume_from_path and not resume_from_path.startswith("/"):
                config.trainer.resume_from_path = os.path.join(amlt_data_dir, resume_from_path)

        if amlt_output_dir and hasattr(config, "trainer"):
            default_local_dir = getattr(config.trainer, "default_local_dir", "")
            if default_local_dir and not default_local_dir.startswith("/"):
                config.trainer.default_local_dir = os.path.join(amlt_output_dir, default_local_dir)

            validation_dir = getattr(config.trainer, "validation_data_dir", "")
            if validation_dir and not validation_dir.startswith("/"):
                config.trainer.validation_data_dir = os.path.join(amlt_output_dir, validation_dir)


def _to_file_list(obj: Any) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return [str(item) for item in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed]
            except (SyntaxError, ValueError):
                pass
        return [stripped]
    return [str(obj)]


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        lowered = val.strip().lower()
        return lowered not in {"0", "false", "no", "off"}
    return bool(val)


def _resolve_filter_cfg(config: DictConfig) -> dict[str, Any]:
    raw_cfg = getattr(config, "filtering", None)
    raw_dict = OmegaConf.to_container(raw_cfg, resolve=True) if raw_cfg is not None else {}
    raw_dict = raw_dict or {}

    target_split = str(raw_dict.get("target_split", "train")).lower()
    target_files_override = raw_dict.get("target_files")
    target_files = _to_file_list(target_files_override)
    if not target_files:
        if target_split == "val":
            target_files = _to_file_list(config.data.val_files)
        else:
            target_files = _to_file_list(config.data.train_files)

    min_sr = float(raw_dict.get("min_success_rate", 0.1))
    max_sr = float(raw_dict.get("max_success_rate", 0.9))
    if min_sr >= max_sr:
        raise ValueError("filtering.min_success_rate must be lower than filtering.max_success_rate")

    write_filtered_dataset_raw = raw_dict.get("write_filtered_dataset", True)
    write_filtered_dataset = _to_bool(write_filtered_dataset_raw)

    output_file = raw_dict.get("output_file")
    if output_file is None:
        fallback = Path(target_files[0]).with_suffix("")
        output_file = str(fallback.with_name(f"{fallback.name}.filtered.parquet"))
    metadata_path = raw_dict.get("metadata_path") or f"{output_file}.success.jsonl"

    filter_cfg = {
        "target_files": target_files,
        "target_split": target_split,
        "min_success_rate": min_sr,
        "max_success_rate": max_sr,
        "rollouts_per_prompt": raw_dict.get("rollouts_per_prompt"),
        "max_samples": int(raw_dict.get("max_samples", -1)),
        "val_batch_size": raw_dict.get("val_batch_size"),
        "output_file": output_file,
        "metadata_path": metadata_path,
        "write_filtered_dataset": write_filtered_dataset,
    }
    return filter_cfg


def _apply_filter_overrides(config: DictConfig, filter_cfg: dict[str, Any]) -> None:
    with open_dict(config):
        config.data.val_files = filter_cfg["target_files"]
        config.data.validation_shuffle = False
        if filter_cfg["val_batch_size"] is not None:
            config.data.val_batch_size = int(filter_cfg["val_batch_size"])
        if filter_cfg["max_samples"] > 0:
            config.data.val_max_samples = int(filter_cfg["max_samples"])
        if filter_cfg["rollouts_per_prompt"] is not None:
            config.actor_rollout_ref.rollout.val_kwargs.n = int(filter_cfg["rollouts_per_prompt"])


def _ray_init_if_needed(config: DictConfig) -> None:
    if ray.is_initialized():
        return

    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

    if config.transfer_queue.enable:
        runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
        runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
        runtime_env_kwargs["env_vars"] = runtime_env_vars

    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))


def _ensure_dataset_has_uid(dataset) -> None:
    if "uid" in dataset.dataframe.column_names:
        return

    # Prefer existing id-like columns if present
    for col in ["id", "sample_id", "row_id"]:
        if col in dataset.dataframe.column_names:
            dataset.dataframe = dataset.dataframe.add_column("uid", [str(x) for x in dataset.dataframe[col]])
            return

    # Try to pull from nested extra_info if it has a stable key like "index" or "id"
    if "extra_info" in dataset.dataframe.column_names:
        extras = dataset.dataframe["extra_info"]
        if extras and isinstance(extras[0], dict):
            for key in ["uid", "id", "index", "sample_id", "row_id"]:
                if all(isinstance(item, dict) and key in item for item in extras):
                    dataset.dataframe = dataset.dataframe.add_column("uid", [str(item[key]) for item in extras])
                    return

    # Fallback: synthesize sequential ids
    total = len(dataset.dataframe)
    uid_values = [f"sample-{idx:08d}" for idx in range(total)]
    dataset.dataframe = dataset.dataframe.add_column("uid", uid_values)


def _build_success_stats(sample_uids: list[Any], metric_values: list[float]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for uid, value in zip(sample_uids, metric_values, strict=True):
        uid_str = str(uid)
        entry = stats.setdefault(uid_str, {"attempts": 0, "successes": 0, "score_sum": 0.0})
        entry["attempts"] += 1
        score_value = float(value)
        entry["score_sum"] += score_value
        # Success is binary: score > 0 means correct answer
        if score_value > 0:
            entry["successes"] += 1

    for uid, entry in stats.items():
        attempts = entry["attempts"]
        entry["success_rate"] = entry["successes"] / attempts if attempts else 0.0
        entry["avg_score"] = entry["score_sum"] / attempts if attempts else 0.0
    return stats


def _build_filtered_index(dataset, stats: dict[str, dict[str, float]], filter_cfg: dict[str, Any]):
    uids = [str(uid) for uid in dataset["uid"]]
    data_sources = dataset["data_source"] if "data_source" in dataset.column_names else [None] * len(uids)

    keep_indices: list[int] = []
    records: list[dict[str, Any]] = []
    dropped_low = 0
    dropped_high = 0

    for idx, uid in tqdm(
        enumerate(uids), total=len(uids), desc="difficulty-filter", unit="sample"
    ):
        entry = stats.get(uid, {"attempts": 0, "successes": 0, "success_rate": 0.0, "avg_score": 0.0})
        rate = entry.get("success_rate", 0.0)
        keep = filter_cfg["min_success_rate"] <= rate <= filter_cfg["max_success_rate"]
        if keep:
            keep_indices.append(idx)
        elif rate < filter_cfg["min_success_rate"]:
            dropped_low += 1
        else:
            dropped_high += 1

        records.append(
            {
                "uid": uid,
                "data_source": data_sources[idx] if idx < len(data_sources) else None,
                "successes": int(entry.get("successes", 0)),
                "attempts": int(entry.get("attempts", 0)),
                "success_rate": rate,
                "avg_score": float(entry.get("avg_score", 0.0)),
                "keep": keep,
            }
        )

    summary = {
        "total": len(uids),
        "kept": len(keep_indices),
        "dropped_low": dropped_low,
        "dropped_high": dropped_high,
    }
    return keep_indices, records, summary


def _write_outputs(dataset, keep_indices, records, filter_cfg):
    if filter_cfg["write_filtered_dataset"]:
        output_path = Path(filter_cfg["output_file"]).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered = dataset.select(keep_indices)
        filtered.to_parquet(str(output_path))
        print(f"[difficulty-filter] wrote filtered dataset -> {output_path}")
    else:
        print("[difficulty-filter] write_filtered_dataset=False, skipping filtered parquet")

    metadata_path = Path(filter_cfg["metadata_path"]).expanduser()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fout:
        for record in records:
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"[difficulty-filter] wrote rollout stats -> {metadata_path}")


def _extract_acc_values(validation_data: dict) -> tuple[list[float], str]:
    extras: dict[str, list] = validation_data["reward_extra_infos_dict"]
    
    # First try to find "acc" key (when compute_score returns a scalar)
    candidate_keys = [key for key in extras.keys() if "acc" in key.lower()]
    for key in candidate_keys:
        values = [float(v) for v in extras[key]]
        return values, key
    
    # Fallback to "score" key (when compute_score returns a dict)
    if "score" in extras:
        values = [float(v) for v in extras["score"]]
        return values, "score"

    available = ", ".join(extras.keys())
    raise RuntimeError(
        "Could not find an accuracy/score metric in reward_extra_infos_dict. "
        f"Ensure the reward function logs accuracy or score (available keys: {available})"
    )


def _instantiate_trainer(config: DictConfig, filter_cfg: dict[str, Any]):
    runner = TaskRunner()
    actor_cls, ray_worker_group_cls = runner.add_actor_rollout_worker(config)
    # For filtering, we don't need critic, reward_model, or ref_policy
    # Only add them if explicitly enabled in config
    if config.critic.get("enable", False):
        runner.add_critic_worker(config)
    if config.reward_model.get("enable", False):
        runner.add_reward_model_worker(config)
    # Reference policy is typically not needed for filtering
    # runner.add_ref_policy_worker(config, actor_cls)
    resource_pool_manager = runner.init_resource_pool_mgr(config)

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(runner.role_worker_mapping),
        use_critic=need_critic(config),
    )

    local_model_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_model_path, trust_remote_code=trust_remote_code, use_fast=True)

    reward_kwargs = config.reward_model.get("reward_kwargs", {})
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs)
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **reward_kwargs)

    from verl.utils.dataset.rl_dataset import collate_fn

    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        is_train=True,
        max_samples=config.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=config.data.get("val_max_samples", -1),
    )

    # Hard clamp for filtering runs to avoid mismatched Hydra overrides.
    max_val_samples = int(filter_cfg.get("max_samples", -1))
    if max_val_samples > 0 and len(val_dataset) > max_val_samples:
        val_dataset.dataframe = val_dataset.dataframe.select(range(max_val_samples))
        print(f"[difficulty-filter] val_max_samples applied: {max_val_samples}/{len(val_dataset)}")
    _ensure_dataset_has_uid(val_dataset)

    train_sampler = create_rl_sampler(config.data, train_dataset)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=runner.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
    )
    trainer.init_workers()
    return trainer, val_dataset


def run_dataset_filter(config: DictConfig, filter_cfg: dict[str, Any]) -> None:
    _ray_init_if_needed(config)
    trainer, eval_dataset = _instantiate_trainer(config, filter_cfg)

    total_prompts = len(eval_dataset)
    rollouts_per_prompt = int(filter_cfg.get("rollouts_per_prompt") or config.actor_rollout_ref.rollout.val_kwargs.n)
    total_rollouts = total_prompts * rollouts_per_prompt
    print(
        f"[difficulty-filter] rollout plan: prompts={total_prompts}, rollouts_per_prompt={rollouts_per_prompt}, total_rollouts={total_rollouts}"
    )

    # Initialize global_steps for validation (normally set in fit())
    trainer.global_steps = 0

    try:
        validation_data = trainer._collect_validation_samples()
    finally:
        ray.shutdown()

    if validation_data is None:
        raise RuntimeError(
            "Validation requires a rule-based reward. Ensure reward_model.enable is False or uses math/code scorers."
        )

    sample_uids = validation_data["sample_uids"]
    metric_values, metric_key = _extract_acc_values(validation_data)
    stats = _build_success_stats(sample_uids, metric_values)

    # Sanity checks: rollout coverage per prompt
    expected_prompts = len(eval_dataset.dataframe)
    expected_rollouts = int(filter_cfg.get("rollouts_per_prompt") or config.actor_rollout_ref.rollout.val_kwargs.n)
    total_attempts = len(sample_uids)
    if expected_prompts > 0 and expected_rollouts > 0:
        expected_total = expected_prompts * expected_rollouts
        if total_attempts != expected_total:
            print(
                f"[difficulty-filter][warn] rollout count mismatch: got {total_attempts}, expected {expected_total} (prompts={expected_prompts}, rollouts_per_prompt={expected_rollouts})"
            )

    # Warn if any prompt has zero attempts (should not happen)
    missing = [uid for uid in eval_dataset.dataframe["uid"] if uid not in stats]
    if missing:
        print(f"[difficulty-filter][warn] {len(missing)} prompts have zero attempts: e.g., {missing[:5]}")

    keep_indices, records, summary = _build_filtered_index(eval_dataset.dataframe, stats, filter_cfg)
    if not keep_indices and filter_cfg["write_filtered_dataset"]:
        raise RuntimeError(
            "All samples were filtered out. Relax filtering.min_success_rate/max_success_rate ranges."
        )

    _write_outputs(eval_dataset.dataframe, keep_indices, records, filter_cfg)
    print(
        "[difficulty-filter] processed={total} kept={kept} low={low} high={high}".format(
            total=summary["total"],
            kept=summary["kept"],
            low=summary["dropped_low"],
            high=summary["dropped_high"],
        )
    )
    if not filter_cfg["write_filtered_dataset"]:
        print("[difficulty-filter] stats-only run complete; no filtered dataset written")
    print(f"[difficulty-filter] metric_used=acc (key={metric_key})")


@hydra.main(config_path="../../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    _maybe_prepend_amlt_paths(config)
    filter_cfg = _resolve_filter_cfg(config)
    _apply_filter_overrides(config, filter_cfg)
    run_dataset_filter(config, filter_cfg)


if __name__ == "__main__":
    main()
