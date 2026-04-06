# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import difflib
import json
import json_repair
import logging
import math
import os
import random
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
from collections import defaultdict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"

#CLEANER: support rollback and negative samples
class RollbackManager:
    """Manages rollback mechanism for tool call errors."""
    
    def __init__(self, enable: bool, max_retries: int, error_patterns: list[str], 
                 save_negative_samples: bool = False, max_negative_samples_per_group: int = 1,
                 is_threshold: Optional[float] = None):
        self.enable = enable
        self.max_retries = max_retries
        self.error_patterns = error_patterns
        self.save_negative_samples = save_negative_samples
        self.max_negative_samples_per_group = max_negative_samples_per_group
        # Plan B: IS-based rollback decision threshold.
        # When set, _handle_rollback computes IS ratio = π_old(R|no_err_ctx) / π_b(R|err_ctx).
        # If IS >= is_threshold → replace (trajectory without error context, near on-policy).
        # If IS < is_threshold  → append (keep full on-policy trajectory including error).
        self.is_threshold = is_threshold
    
    def can_retry(self, retry_counts: dict[str, int], position_key: str) -> bool:
        """Check if retry is allowed at this position."""
        return retry_counts.get(position_key, 0) < self.max_retries
    
    def increment_retry(self, retry_counts: dict[str, int], position_key: str) -> int:
        """Increment retry count and return new count."""
        current = retry_counts.get(position_key, 0)
        retry_counts[position_key] = current + 1
        return retry_counts[position_key]
    
    def format_error_feedback(self, error_messages: list[str], error_types: list[str]) -> str:
        """Format error feedback for LLM."""
        # feedback = "The previous tool call(s) failed with the following error:\n"
        # feedback += f"error type:{error_types[-1]}, error message:{error_messages[-1]}\n"
        # if error_types[-1] == "worker_timeout":
        #     feedback += "The current algorithm complexity may be too high."
        # feedback += "\nPlease correct the error and generate a new tool call."
        feedback = error_messages[-1]  # only return normal error message is fine
        return feedback
    
    def create_checkpoint(self, agent_data: "AgentData") -> dict[str, Any]:
        """Create a checkpoint of current agent state."""
        return {
            "prompt_ids": list(agent_data.prompt_ids),
            "response_ids": agent_data.response_ids,
            "response_mask": list(agent_data.response_mask),
            "response_logprobs": list(agent_data.response_logprobs) if agent_data.response_logprobs is not None else None,
            "rollback_response_mask": list(agent_data.rollback_response_mask),
            "messages": copy.deepcopy(agent_data.messages),
            "image_data": agent_data.image_data,
            "assistant_turns": agent_data.assistant_turns,
            "user_turns": agent_data.user_turns,
        }
    
    def restore_checkpoint(self, agent_data: "AgentData", checkpoint: dict[str, Any]):
        """Restore agent state from checkpoint."""
        agent_data.prompt_ids = checkpoint["prompt_ids"]
        agent_data.response_ids = checkpoint["response_ids"]
        agent_data.response_mask = checkpoint["response_mask"]
        agent_data.response_logprobs = checkpoint["response_logprobs"]
        agent_data.rollback_response_mask = checkpoint.get("rollback_response_mask", [])
        agent_data.messages = checkpoint["messages"]
        agent_data.image_data = checkpoint["image_data"]
        agent_data.assistant_turns = checkpoint["assistant_turns"]
        agent_data.user_turns = checkpoint["user_turns"]


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: Optional[list[float]] = None
        self.rollback_response_mask: list[int] = []  # 1 for tokens replaced by rollback, 0 otherwise
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        #CLEANER: Tool call statistics per trajectory
        self.tool_call_total = 0
        self.tool_call_success = 0
        self.tool_failure_reasons: defaultdict[str, int] = defaultdict(int) # use defaultdict to avoid key error
        
        #CLEANER: First attempt statistics (excluding rollback retries)
        self.first_attempt_total = 0
        self.first_attempt_success = 0
        
        #CLEANER: Global rollback statistics per trajectory
        self.global_rollback_triggered = 0
        self.global_rollback_recovered = 0
        self.global_rollback_failed = 0
        
        #CLEANER: Rollback strategy statistics
        self.rollback_append_count = 0   # Plan B: append path (IS < threshold)
        self.rollback_replace_count = 0  # Plan B: replace path (IS >= threshold)

        #CLEANER: support rollback
        self.retry_counts: dict[str, int] = defaultdict(int)
        self.total_rollbacks = 0
        self.disable_rollback_after_max_retry = False
        self.rollback_recovered_turns: set[str] = set()
        # Deferred rollback: tool_call_idx = 第几次「独立」tool call（回滚重试、第一次错后的修正都不算新的独立 call）。
        # tool_call_attempt_count[idx] = 该 idx 下第几次尝试；count > 1 时才启动回滚。
        self.tool_call_idx: int = 0
        self.tool_call_attempt_count: dict[int, int] = defaultdict(int)
        self._next_tool_call_is_correction: bool = False  # 免回滚后下一轮是「修正」而非新独立 call，不递增 tool_call_idx
        # 追踪首次错误状态（用于判断是首次调用还是修复尝试）
        self.first_error_tool_idx: Optional[int] = None  # 首次出错的 tool_call_idx
        self.waiting_for_correction: bool = False  # 是否在等待修正结果
        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        
        #CLEANER: Negative samples for failed tool calls
        self.negative_samples: list[AgentLoopOutput] = []  # Store failed trajectories as AgentLoopOutput instances
        self.negative_samples_count = 0  # Track number of negative samples collected
        self.pending_negative_samples: dict[str, AgentLoopOutput] = {}  # Defer persistence until rollback succeeds
        self.saved_tool_call_token_range: Optional[tuple[int, int]] = None  # Tool call range when negative sample was saved

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        # Initialize tools from config file
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        self.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format

        #CLEANER: Initialize rollback manager
        enable_rollback = config.actor_rollout_ref.rollout.multi_turn.get("enable_tool_rollback", False)
        max_retries = config.actor_rollout_ref.rollout.multi_turn.get("max_tool_retries", 3)
        error_patterns = config.actor_rollout_ref.rollout.multi_turn.get(
            "rollback_on_errors",
            ["ImportError", "ModuleNotFoundError", "SyntaxError", "IndexError", "IndentationError", "NameError", "TypeError", "worker_timeout","NotImplementedError","ValueError","ZeroDivisionError"],
        )
        #CLEANER: config setting for negative samples
        save_negative_samples = config.actor_rollout_ref.rollout.multi_turn.get("save_negative_samples", False)
        max_negative_samples_per_group = config.actor_rollout_ref.rollout.multi_turn.get("max_negative_samples_per_group", 1)
        # Plan B: IS-based rollback decision threshold (None = disabled, use replace always)
        rollback_is_threshold = config.actor_rollout_ref.rollout.multi_turn.get("rollback_is_threshold", None)
        self.rollback_manager = RollbackManager(enable_rollback, max_retries, error_patterns, save_negative_samples, max_negative_samples_per_group, is_threshold=rollback_is_threshold)
        #CLEANER: Probability of enabling rollback per sample (0.0-1.0, default 1.0 means all samples have rollback)
        self.rollback_probability = config.actor_rollout_ref.rollout.multi_turn.get("rollback_probability", 1.0)
        #CLEANER: dynamic running control for negative sampling, only when both true, negative samples are collected 
        self.negative_sampling_enabled = kwargs.get("negative_sampling_enabled", True)
        self.is_validation = kwargs.get("validate", False)  # Passed from agent_loop.py during instantiation

        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)

        # Initialize interactions from config file
        self.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        
        #CLEANER: Randomly disable rollback for some samples based on rollback_probability
        # If random value >= probability, disable rollback for this sample from the start
        if random.random() >= self.rollback_probability:
            agent_data.disable_rollback_after_max_retry = True

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs is not None else None,
            rollback_response_mask=agent_data.rollback_response_mask[: self.response_length],
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
            is_negative=False,  # Main trajectory is always positive
            negative_parent_index=-1,
            negative_sample_id=None,  # Only negative samples have IDs
            paired_negative_id=None,  # Will be set below if negative samples exist
            #CLEANER: DPO paired training fields
            tool_call_token_range=agent_data.saved_tool_call_token_range if agent_data.negative_samples else None,
        )
        #CLEANER: add tool call statistics to extra_fields
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "tool_call_total": agent_data.tool_call_total,
                "tool_call_success": agent_data.tool_call_success,
                "tool_failure_reasons": agent_data.tool_failure_reasons,
                "first_attempt_total": agent_data.first_attempt_total,
                "first_attempt_success": agent_data.first_attempt_success,
                "global_rollback_triggered": agent_data.global_rollback_triggered,
                "global_rollback_recovered": agent_data.global_rollback_recovered,
                "global_rollback_failed": agent_data.global_rollback_failed,
                "rollback_append_count": agent_data.rollback_append_count,
                "rollback_replace_count": agent_data.rollback_replace_count,
            }
        )
        
        #CLEANER: Attach paired negative sample to positive sample (if any)
        if agent_data.negative_samples:
            # Assuming max_negative_samples_per_group=1, only one negative sample
            output.paired_negative_id = agent_data.negative_samples[0].negative_sample_id
            return [output] + agent_data.negative_samples
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        agent_data.rollback_response_mask += [0] * len(agent_data.response_ids)
        if output.log_probs:
            if agent_data.response_logprobs is None:
                agent_data.response_logprobs = []
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], tool_position_key: Optional[str] = None
    ) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses with rollback support."""
        #CLEANER: safeguard for rollback, though should not happen
        agent_data.total_rollbacks += 1
        MAX_TOOL_ATTEMPTS_BEFORE_DISABLE = 30

        if agent_data.total_rollbacks > MAX_TOOL_ATTEMPTS_BEFORE_DISABLE and not agent_data.disable_rollback_after_max_retry:
            logger.warning(
                f"⚠️ Total tool attempts ({agent_data.total_rollbacks}) exceeded {MAX_TOOL_ATTEMPTS_BEFORE_DISABLE}. "
                f"Disabling rollback to prevent infinite loops."
            )
            agent_data.disable_rollback_after_max_retry = True
        
        # New tool call vs rollback retry: only when we're not given an explicit key do we have a fresh tool call.
        if tool_position_key is None:
            tool_position_key = f"turn_{agent_data.assistant_turns}"
            agent_data.tool_call_idx += 1
        
        # Check if this is the first attempt at this position (not a retry)
        is_first_attempt = (agent_data.retry_counts.get(tool_position_key, 0) == 0)
        
        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data, is_first_attempt=is_first_attempt))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
        
        # Detect errors
        error_messages, error_types = self._detect_errors(responses, tool_position_key, agent_data)
        
        # Handle rollback if enabled
        if self.rollback_manager.enable and (not agent_data.disable_rollback_after_max_retry):
            if error_messages:
                retry_count = agent_data.retry_counts.get(tool_position_key, 0)

                # Plan B: every error immediately triggers rollback (no 1-tolerance).
                should_trigger_rollback = True

                if should_trigger_rollback:
                    # Save/update negative sample on every rollback-triggering failure (if enabled)
                    if (self.negative_sampling_enabled and
                        not self.is_validation and
                        self.rollback_manager.save_negative_samples and
                        agent_data.negative_samples_count < self.rollback_manager.max_negative_samples_per_group):
                        negative_sample = self._create_negative_sample(agent_data)
                        agent_data.pending_negative_samples[tool_position_key] = negative_sample

                    # Record rollback trigger on the first rollback attempt at this position
                    if retry_count == 0:
                        agent_data.global_rollback_triggered += 1

                    # Check retry limit
                    exceeded_retries = not self.rollback_manager.can_retry(agent_data.retry_counts, tool_position_key)
                    if exceeded_retries or agent_data.disable_rollback_after_max_retry:
                        # Drop pending negative sample on final failure
                        agent_data.pending_negative_samples.pop(tool_position_key, None)
                        if tool_position_key not in agent_data.rollback_recovered_turns:
                            agent_data.global_rollback_failed += 1
                        return await self._handle_max_retry_exceeded(
                            agent_data, responses, error_messages, error_types, tool_call_names, sampling_params
                        )

                    # Create checkpoint and handle rollback
                    checkpoint = self.rollback_manager.create_checkpoint(agent_data)
                    rollback_result = await self._handle_rollback(
                        agent_data, checkpoint, tool_position_key, error_messages, error_types, sampling_params
                    )
                    if rollback_result is not None:
                        return rollback_result
            else:
                # Success after rollback-retry - persist negative sample
                retry_count = agent_data.retry_counts.get(tool_position_key, 0)
                if retry_count > 0:
                    pending_sample = agent_data.pending_negative_samples.pop(tool_position_key, None)
                    if (pending_sample is not None and
                        agent_data.negative_samples_count < self.rollback_manager.max_negative_samples_per_group):
                        # Record tool_call_token_range at the moment of saving negative sample (only once)
                        if not agent_data.negative_samples:
                            agent_data.saved_tool_call_token_range = self._calculate_tool_call_range(agent_data)
                        agent_data.negative_samples.append(pending_sample)
                        agent_data.negative_samples_count += 1

                    agent_data.rollback_recovered_turns.add(tool_position_key)
                    agent_data.global_rollback_recovered += 1
        
        # Process tool responses normally
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        agent_data.rollback_response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs is not None:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        agent_data.rollback_response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs is not None:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData, is_first_attempt: bool = False
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json_repair.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            #CLEANER: add detailed response for format error
            logger.warning(f"tool call error: {e}")
            # Check if it's a format-related error (AttributeError, TypeError, KeyError, json parsing)
            if isinstance(e, (AttributeError, TypeError, KeyError)) or "'str' object has no attribute" in str(e).lower():
                error_response = (
                    "Tool call failure: tool call format is wrong, please make sure to generate correct "
                    "json-format tool call arguments."
                )
            else:
                error_response = "Tool call failure."
            self._record_tool_attempt(agent_data, success=False, failure_reason="tool_call_format_error", is_first_attempt=is_first_attempt)
            return (
                ToolResponse(
                    text=error_response,
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        #CLEANER: detect error type and record tool call statistics
        has_error, error_type = self._detect_error_from_text(tool_response_text)
        self._record_tool_attempt(agent_data, success=not has_error, failure_reason=error_type, is_first_attempt=is_first_attempt)
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map

    #CLEANER: record tool call statistics
    def _record_tool_attempt(self, agent_data: AgentData, success: bool, failure_reason: Optional[str] = None, is_first_attempt: bool = False):
        """Record that a tool call was attempted for this trajectory.
        
        Args:
            agent_data: Agent data containing statistics
            success: Whether the tool call succeeded
            failure_reason: Reason for failure (if any)
            is_first_attempt: Whether this is the first attempt (not a retry after rollback)
        """
        agent_data.tool_call_total += 1
        if success:
            agent_data.tool_call_success += 1
        else:
            reason_key = failure_reason or "unknown_failure"
            agent_data.tool_failure_reasons[reason_key] += 1
        
        # Record first attempt statistics separately
        if is_first_attempt:
            agent_data.first_attempt_total += 1
            if success:
                agent_data.first_attempt_success += 1
    
    #CLEANER: error detection function
    def _detect_error_from_text(self, text: str) -> tuple[bool, Optional[str]]:
        """Detect error from tool response text.
        
        Returns:
            tuple[bool, Optional[str]]: (has_error, error_type)
                - has_error: True if error detected, False otherwise
                - error_type: matched error pattern or None
        """
        if not text:
            return False, None
        
        # Success case
        if "Tool call success" in text:
            return False, None
        
        # Check against rollback error patterns
        for pattern in self.rollback_manager.error_patterns:
            if pattern in text:
                return True, pattern
        
        return True, "unknown_error"
    
    def _detect_errors(
        self, responses: list[tuple], tool_position_key: str, agent_data: AgentData
    ) -> tuple[list[str], list[str]]:
        """Detect rollback-triggering errors in tool responses.
        
        Returns:
            tuple[list[str], list[str]]: (error_messages, error_types)
        """
        error_messages = []
        error_types = []
        
        for tool_response, _, _ in responses:
            error_text = tool_response.text or ""
            has_error, error_type = self._detect_error_from_text(error_text)
            if has_error and error_type:
                error_messages.append(error_text)
                error_types.append(error_type)
        
        return error_messages, error_types
    
    #CLEANER: below functions are all for rollback handling
    async def _handle_rollback(
        self, agent_data: AgentData, checkpoint: dict[str, Any], tool_position_key: str,
        error_messages: list[str], error_types: list[str], sampling_params: dict[str, Any]
    ) -> Optional[AgentState]:
        # print("⚠️ [ROLLBACK] Initiating rollback process...")
        """Handle the rollback process."""
        if not error_messages:
            return None
        
        self.rollback_manager.increment_retry(agent_data.retry_counts, tool_position_key)
        
        # Step 1: Append error feedback as tool response
        error_feedback = self.rollback_manager.format_error_feedback(error_messages, error_types)
        # print(f"\n[ROLLBACK] ▶ trigger  key={tool_position_key}  "
        #       f"retry={agent_data.retry_counts.get(tool_position_key, 0)}  "
        #       f"error_type={error_types[-1] if error_types else '?'}")
        # print(f"[ROLLBACK]   error_msg (tail 200): ...{error_feedback[-200:]!r}")
        error_message = {"role": "tool", "content": error_feedback}
        agent_data.messages.append(error_message)
        
        # Step 2: Encode error feedback
        error_prompt_ids = await self._encode_error_feedback(agent_data, error_message)
        agent_data.prompt_ids += error_prompt_ids
        agent_data.response_mask += [0] * len(error_prompt_ids)
        agent_data.rollback_response_mask += [0] * len(error_prompt_ids)
        
        logprob_offset = None
        if agent_data.response_logprobs is not None:
            agent_data.response_logprobs += [0.0] * len(error_prompt_ids)
            logprob_offset = len(agent_data.response_logprobs)
        
        # Step 3: Regenerate tool calls
        new_state = await self._handle_generating_state(agent_data, sampling_params, ignore_termination=True)
        
        if new_state == AgentState.TERMINATED and agent_data.tool_calls:
            if tool_position_key not in agent_data.rollback_recovered_turns:
                agent_data.global_rollback_failed += 1
            self.rollback_manager.restore_checkpoint(agent_data, checkpoint)
            return AgentState.TERMINATED
        
        if not agent_data.tool_calls:
            if tool_position_key not in agent_data.rollback_recovered_turns:
                agent_data.global_rollback_failed += 1
            agent_data.disable_rollback_after_max_retry = True
            return new_state
        
        # Step 4: Prepare checkpoint with regenerated tool call
        new_response_ids = list(agent_data.response_ids)
        if not new_response_ids:
            self.rollback_manager.restore_checkpoint(agent_data, checkpoint)
            return await self._handle_processing_tools_state(agent_data, sampling_params, tool_position_key)
        
        new_response_logprobs: Optional[list[float]] = None
        if logprob_offset is not None and agent_data.response_logprobs is not None:
            new_response_logprobs = agent_data.response_logprobs[logprob_offset:]
        
        # Plan B: IS-based replace vs. append decision
        if self.rollback_manager.is_threshold is not None:
            use_append = await self._decide_rollback_append(
                agent_data, checkpoint, new_response_ids, new_response_logprobs
            )
            if use_append:
                # Append path: trajectory already contains [original | old_resp | error | new_resp].
                # This is fully on-policy for new_resp.  Just continue executing new tool calls.
                agent_data.rollback_append_count += 1
                # print(f"[ROLLBACK] ✅ APPEND path → trajectory length={len(agent_data.prompt_ids)} tokens "
                #       f"(on-policy, keeping error context)")
                return await self._handle_processing_tools_state(agent_data, sampling_params, tool_position_key)
            else:
                agent_data.rollback_replace_count += 1
                # print(f"[ROLLBACK] ↩  REPLACE path → will overwrite old response + error from trajectory")
        
        new_assistant_message: Optional[dict[str, Any]] = None
        if self.interaction_config_file and agent_data.messages:
            last_message = agent_data.messages[-1]
            if last_message.get("role") == "assistant":
                new_assistant_message = copy.deepcopy(last_message)
        
        self._overwrite_last_assistant_turn(checkpoint, new_response_ids, new_response_logprobs, new_assistant_message, error_message, agent_data)
        
        # Step 5: Restore checkpoint
        self.rollback_manager.restore_checkpoint(agent_data, checkpoint)
        
        # Step 6: Recursive retry
        return await self._handle_processing_tools_state(agent_data, sampling_params, tool_position_key)
    
    async def _decide_rollback_append(
        self,
        agent_data: AgentData,
        checkpoint: dict[str, Any],
        new_response_ids: list[int],
        new_response_logprobs: Optional[list[float]],
    ) -> bool:
        """Plan B: decide whether to append (keep full on-policy trajectory) or replace.

        Computes IS ratio = π_old(R | no_error_ctx) / π_b(R | with_error_ctx) where:
          - no_error_ctx = checkpoint prompt before the old response (no error feedback)
          - with_error_ctx = full prompt including old response + error feedback (at generation time)

        Returns True  → append path (IS < threshold, keep full trajectory, on-policy).
                False → replace path (IS ≥ threshold, trajectory close to no-error distribution).
        """
        # no_error_ctx: everything before old_response in checkpoint
        old_response_ids = list(checkpoint.get("response_ids") or [])
        old_response_len = len(old_response_ids)
        if old_response_len > 0:
            no_error_ctx = checkpoint["prompt_ids"][:-old_response_len]
        else:
            no_error_ctx = list(checkpoint["prompt_ids"])

        # with_error_ctx: prompt at the moment new_response was generated
        # = agent_data.prompt_ids[:-len(new_response_ids)]
        with_error_ctx = agent_data.prompt_ids[:-len(new_response_ids)]

        # error_feedback_ids: tokens between checkpoint prompt end and new_response start
        # = with_error_ctx[len(checkpoint["prompt_ids"]):]
        error_feedback_ids = with_error_ctx[len(checkpoint["prompt_ids"]):]

        # ── helpers ───────────────────────────────────────────────────────────
        def _tail(ids: list[int], n: int = 200) -> str:
            """Decode last n chars of a token list for context preview."""
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            return text[-n:] if len(text) > n else text

        def _decode(ids: list[int]) -> str:
            return self.tokenizer.decode(ids, skip_special_tokens=False)

        def _logprob_summary(lps: list[float], ids: list[int], n_head: int = 6, n_tail: int = 4) -> str:
            """Print first/last few (token, logprob) pairs."""
            tokens = [self.tokenizer.decode([t], skip_special_tokens=False) for t in ids]
            pairs = list(zip(tokens, lps))
            shown = pairs[:n_head]
            if len(pairs) > n_head + n_tail:
                shown.append(("...", float("nan")))
            if len(pairs) > n_head:
                shown.extend(pairs[-n_tail:])
            lines = []
            for tok, lp in shown:
                tok_repr = repr(tok)
                lines.append(f"  {tok_repr:>20s}  {lp:+.4f}" if not math.isnan(lp) else f"  {'...':>20s}")
            return "\n".join(lines)

        # ── DEBUG header ──────────────────────────────────────────────────────
        # retry_key = 'turn_' + str(checkpoint.get('assistant_turns', '?'))
        # print(f"\n{'='*72}")
        # print(f"[PLAN-B IS DEBUG] request={agent_data.request_id[:12]}  "
        #       f"retry={agent_data.retry_counts.get(retry_key, '?')}")
        # print(f"  preceding ctx : {len(no_error_ctx):5d} tokens  tail → ...{_tail(no_error_ctx, 200)!r}")
        # print(f"{'─'*72}")
        # print(f"[old response  | {old_response_len} tokens]")
        # print(_decode(old_response_ids))
        # print(f"{'─'*72}")
        # print(f"[error feedback| {len(error_feedback_ids)} tokens]")
        # print(_decode(error_feedback_ids))
        # print(f"{'─'*72}")
        # print(f"[new response  | {len(new_response_ids)} tokens]")
        # print(_decode(new_response_ids))
        # print(f"{'─'*72}")
        # ──────────────────────────────────────────────────────────────────────

        try:
            # π_old(R | no_error_ctx) via scoring
            no_error_logprobs = await self.server_manager.score(
                agent_data.request_id,
                prompt_ids=no_error_ctx,
                response_ids=new_response_ids,
            )
            no_error_sum = sum(no_error_logprobs)

            # π_b(R | with_error_ctx): prefer generation logprobs (free), else score separately
            from_generation = (
                new_response_logprobs is not None
                and len(new_response_logprobs) == len(new_response_ids)
            )
            if from_generation:
                with_error_logprobs_used = new_response_logprobs
                with_error_sum = sum(new_response_logprobs)
                pi_b_source = "generation logprobs (calculate_log_probs=True)"
            else:
                with_error_logprobs_used = await self.server_manager.score(
                    agent_data.request_id,
                    prompt_ids=with_error_ctx,
                    response_ids=new_response_ids,
                )
                with_error_sum = sum(with_error_logprobs_used)
                pi_b_source = "score() call (calculate_log_probs=False)"

            n = max(len(new_response_ids), 1)
            log_is_sum = no_error_sum - with_error_sum
            # Sequence-level geometric-mean IS: normalize by response length so the
            # ratio is per-token scale and independent of sequence length.
            log_is_per_tok = log_is_sum / n
            # Clamp per-token log-IS to ±3 (exp(±3) ≈ 0.05 / 20) to avoid overflow
            log_is_clamped = max(min(log_is_per_tok, 3.0), -3.0)
            is_ratio = math.exp(log_is_clamped)  # IS_geom = (IS_product)^(1/n)
            decision = (
                "APPEND  (keep full on-policy trajectory)"
                if is_ratio < self.rollback_manager.is_threshold
                else "REPLACE (trajectory near no-error dist)"
            )

            # ── DEBUG per-token breakdown ──────────────────────────────────
            # print(f"[π_old | no_error_ctx]  sum={no_error_sum:.4f}  avg/tok={no_error_sum/n:.4f}")
            # print(_logprob_summary(no_error_logprobs, new_response_ids))
            # print(f"[π_b   | with_error_ctx]  sum={with_error_sum:.4f}  avg/tok={with_error_sum/n:.4f}"
            #       f"  source: {pi_b_source}")
            # print(_logprob_summary(with_error_logprobs_used, new_response_ids))
            # print(f"{'─'*72}")
            # print(f"log_IS (sum)     = {log_is_sum:+.4f}")
            # print(f"log_IS (per-tok) = {log_is_per_tok:+.6f}  (clamped → {log_is_clamped:+.6f})")
            # print(f"IS_geom          = {is_ratio:.6f}   (= geometric-mean per-token IS ratio)")
            # print(f"threshold        = {self.rollback_manager.is_threshold:.6f}")
            # print(f"decision         → {decision}")
            # print(f"{'='*72}\n")
            # ──────────────────────────────────────────────────────────────

            return is_ratio < self.rollback_manager.is_threshold

        except Exception as e:
            # print(f"[PLAN-B IS DEBUG] score() FAILED: {e!r}  → fallback to REPLACE")
            # print(f"{'='*72}\n")
            # Scoring failure: fall back to replace path to avoid silently broken trajectories
            logger.warning(f"[Plan-B IS] score() failed ({e}), falling back to replace path")
            return False

    def _overwrite_last_assistant_turn(
        self, checkpoint: dict[str, Any], new_response_ids: list[int],
        new_response_logprobs: Optional[list[float]], new_assistant_message: Optional[dict[str, Any]],
        error_message: Optional[str], agent_data: Optional[AgentData] = None
    ) -> None:
        """Replace the last assistant turn entirely with the new response (Plan B: always full replacement)."""
        old_response_ids = list(checkpoint.get("response_ids") or [])

        # Plan B always does a full turn replacement (no partial tool-call-only path).
        self._replace_full_turn(checkpoint, old_response_ids, new_response_ids, new_response_logprobs)
        if agent_data is not None:
            pass  # replace count already tracked at _handle_rollback call site

        # Update assistant message in checkpoint if present
        if new_assistant_message is not None:
            new_text = self._decode_response_text(new_response_ids)
            new_assistant_message["content"] = new_text
            for idx in range(len(checkpoint["messages"]) - 1, -1, -1):
                message = checkpoint["messages"][idx]
                if message.get("role") == "assistant":
                    checkpoint["messages"][idx] = new_assistant_message
                    break
        old_text = self._decode_response_text(old_response_ids) if old_response_ids else ""
        new_text = self._decode_response_text(checkpoint["response_ids"]) if checkpoint.get("response_ids") else ""
        # print(f"\n{'='*70}")
        # print("📝 [REPLACE SUMMARY] old → new")
        # print(f"{'-'*70}")
        # print(f"Old response ({len(old_response_ids)} tokens):\n{old_text}")
        # print(f"{'-'*70}")
        # print(f"New response ({len(checkpoint['response_ids'])} tokens):\n{new_text}")
        # print(f"{'='*70}\n")

    def _replace_full_turn(
        self, checkpoint: dict[str, Any], old_response_ids: list[int],
        new_response_ids: list[int], new_response_logprobs: Optional[list[float]]
    ) -> None:
        """Replace entire assistant turn (used for low similarity or failed segmentation)."""
        old_response_len = len(old_response_ids)
        
        # Remove old response
        if old_response_len:
            checkpoint["prompt_ids"] = checkpoint["prompt_ids"][:-old_response_len]
            checkpoint["response_mask"] = checkpoint["response_mask"][:-old_response_len]
            if checkpoint.get("response_logprobs"):
                checkpoint["response_logprobs"] = checkpoint["response_logprobs"][:-old_response_len]
            checkpoint["rollback_response_mask"] = checkpoint["rollback_response_mask"][:-old_response_len]
        
        # Add new response
        checkpoint["prompt_ids"].extend(new_response_ids)
        checkpoint["response_mask"].extend([1] * len(new_response_ids))
        checkpoint["response_ids"] = list(new_response_ids)
        checkpoint["rollback_response_mask"].extend([1] * len(new_response_ids))
        
        # Handle logprobs
        if checkpoint.get("response_logprobs"):
            # Only process logprobs if checkpoint originally has logprobs
            logs_to_append = list(new_response_logprobs) if new_response_logprobs else [0.0] * len(new_response_ids)
            if len(logs_to_append) != len(new_response_ids):
                if len(logs_to_append) < len(new_response_ids):
                    logs_to_append.extend([0.0] * (len(new_response_ids) - len(logs_to_append)))
                else:
                    logs_to_append = logs_to_append[:len(new_response_ids)]
            checkpoint["response_logprobs"].extend(logs_to_append)
        # If checkpoint has no logprobs (None), keep it that way
    
    def _split_tool_call_segment(self, token_ids: list[int]) -> Optional[dict[str, Any]]:
        """Locate tool-call tokens and split prefix vs call at token boundaries."""
        if not token_ids:
            return None
        
        # Find the token-level boundary where tool call starts
        tool_call_token_idx = self._find_tool_call_token_boundary(token_ids)
        if tool_call_token_idx is None:
            return None
        
        # Split at token level (guaranteed to be reversible)
        prefix_ids = token_ids[:tool_call_token_idx]
        call_ids = token_ids[tool_call_token_idx:]
        
        # Decode for display/comparison purposes
        prefix_text = self._decode_response_text(prefix_ids)
        call_text = self._decode_response_text(call_ids)
        
        return {
            "prefix_ids": prefix_ids,
            "call_ids": call_ids,
            "prefix_text": prefix_text,
            "call_text": call_text,
        }
    
    def _find_tool_call_token_boundary(self, token_ids: list[int]) -> Optional[int]:
        """Find the token index where tool call starts by iteratively decoding."""
        if not token_ids:
            return None
        
        # Try to find tool call marker by progressively decoding from start
        for i in range(1, len(token_ids) + 1):
            partial_text = self._decode_response_text(token_ids[:i])
            
            # Check if we've found a tool call marker
            if self._contains_tool_call_marker(partial_text):
                # Found the marker, now find exact boundary
                # Search backwards to find where marker actually starts
                for j in range(i, 0, -1):
                    text_before = self._decode_response_text(token_ids[:j])
                    if not self._contains_tool_call_marker(text_before):
                        # j is the last position without marker, so j+1 is where it starts
                        return j
                # Edge case: marker starts at position 0
                return 0
        
        return None
    
    def _contains_tool_call_marker(self, text: str) -> bool:
        """Check if text contains any tool call marker."""
        if not text:
            return False
        
        # Standard markers
        for marker in ("<tool_call>", "<tool_call>\n", "<tool_call>\r", "<tool_call>\r\n"):
            if marker in text:
                return True
        
        # Hermes format
        if self.tool_parser_name == "hermes":
            marker = getattr(self.tool_parser, "tool_call_start_token", "<tool_call>")
            if marker in text:
                return True
        
        # GPT-OSS format
        if self.tool_parser_name == "gpt-oss":
            if " to=functions." in text or "<|constrain|>json" in text:
                if "<|start|>assistant" in text or "<|start|>" in text:
                    return True
        
        return False
    
    def _decode_response_text(self, token_ids: list[int]) -> str:
        """Decode tokens without dropping special markers."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def _encode_response_text(self, text: str) -> list[int]:
        """Encode text without adding specials."""
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    async def _encode_error_feedback(self, agent_data: AgentData, error_message: dict[str, Any]) -> list[int]:
        """Encode error feedback message to token ids."""
        if self.processor is not None:
            raw_error_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    [error_message], add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                ),
            )
            model_inputs = self.processor(text=[raw_error_prompt], images=None, return_tensors="pt")
            return model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            error_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [error_message], add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            return error_prompt_ids[len(self.system_prompt):]
    
    async def _handle_max_retry_exceeded(
        self, agent_data: AgentData, responses: list[tuple], error_messages: list[str],
        error_types: list[str], tool_call_names: list[str], sampling_params: dict[str, Any]
    ) -> AgentState:
        """Handle max retry exceeded: notify model instead of direct termination."""
        retry_count = agent_data.retry_counts.get(f"turn_{agent_data.assistant_turns}", 0)
        if retry_count >=2:
            failure_message = (
                f"{error_messages[-1]}\n\n"
                f"Note: This approach has been attempted {retry_count} times with the same type of error. "
                f"Consider trying a different algorithm, method, or approach to solve this problem."
            )
        else:
            failure_message = error_messages[-1]
        tool_error_message = {"role": "tool", "content": failure_message}
        agent_data.disable_rollback_after_max_retry = True
        
        # Add tool error message only
        agent_data.messages.append(tool_error_message)
        
        # Encode tool error message
        if self.processor is not None:
            raw_notification = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    [tool_error_message], 
                    add_generation_prompt=True, 
                    tokenize=False, 
                    **self.apply_chat_template_kwargs
                ),
            )
            model_inputs = self.processor(text=[raw_notification], images=None, return_tensors="pt")
            notification_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            notification_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [tool_error_message], 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    **self.apply_chat_template_kwargs
                ),
            )
            notification_ids = notification_ids[len(self.system_prompt):]
        
        # Update agent state
        agent_data.prompt_ids += notification_ids
        agent_data.response_mask += [0] * len(notification_ids)
        if agent_data.response_logprobs is not None:
            agent_data.response_logprobs += [0.0] * len(notification_ids)
        agent_data.user_turns += 1
        
        # Check length limit
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        
        return AgentState.GENERATING
    
    def _create_negative_sample(
        self,
        agent_data: AgentData,
    ) -> AgentLoopOutput:
        """Create a fully specified AgentLoopOutput for a failed trajectory."""
        
        prompt_ids_snapshot = list(agent_data.prompt_ids)
        response_mask = list(agent_data.response_mask)
        
        # Negative samples should backprop through:
        # 1. The tool-call tokens (not reasoning prefix)
        # 2. Previous turn's ```python code blocks (if any)
        if response_mask:
            focused_mask = [0] * len(response_mask)
            # Step 1: Mask only the tool call part (not reasoning prefix)
            if agent_data.response_ids:
                segment = self._split_tool_call_segment(agent_data.response_ids)
                if segment and segment.get("call_ids"):
                    # Calculate where the tool call starts in the full response_mask
                    assistant_token_count = len(agent_data.response_ids)
                    tool_call_token_count = len(segment["call_ids"])
                    tool_call_start = len(response_mask) - tool_call_token_count
                    
                    # Mask only the tool call tokens
                    if tool_call_start >= 0:
                        focused_mask[tool_call_start:] = [1] * tool_call_token_count
                else:
                    # Fallback: if can't split, mask all assistant tokens
                    assistant_token_count = len(agent_data.response_ids)
                    start = max(0, len(response_mask) - assistant_token_count)
                    focused_mask[start:] = response_mask[start:]
            
            # Step 2: Removed - only penalize current turn's tool call, not previous code blocks
            # focused_mask = self._mask_previous_code_blocks(
            #     prompt_ids_snapshot, focused_mask, agent_data.response_ids
            # )
            
            response_mask = focused_mask
        
        response_length = len(response_mask)
        if response_length > len(prompt_ids_snapshot):
            response_length = len(prompt_ids_snapshot)
        prompt_prefix = prompt_ids_snapshot[: len(prompt_ids_snapshot) - response_length] if response_length else prompt_ids_snapshot
        response_ids = prompt_ids_snapshot[-response_length:] if response_length else []
        
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        extra_fields = {
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
            "tool_call_total": agent_data.tool_call_total,
            "tool_call_success": agent_data.tool_call_success,
            "tool_failure_reasons": agent_data.tool_failure_reasons,
            "first_attempt_total": agent_data.first_attempt_total,
            "first_attempt_success": agent_data.first_attempt_success,
            "global_rollback_triggered": 0,
            "global_rollback_recovered": 0,
            "global_rollback_failed": 0,
            "rollback_append_count": 0,
            "rollback_replace_count": 0,
        }
        negative_sample_id = str(uuid4())
        return AgentLoopOutput(
            prompt_ids=prompt_prefix,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs is not None
                else None
            ),
            reward_score=None,  # Let negative samples go through normal reward calculation
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields=extra_fields,
            is_negative=True,
            negative_parent_index=-1,  # Will be set by caller
            negative_sample_id=negative_sample_id,  # Unique ID for tracking
            paired_negative_id=None,  # only positive has this
            #CLEANER: DPO paired training fields (must match positive sample structure)
            tool_call_token_range=self._calculate_tool_call_range(agent_data),
        )
    
    def _calculate_tool_call_range(self, agent_data: AgentData) -> Optional[tuple[int, int]]:
        """Calculate the token range of the last tool call in response for DPO masking.
        
        Returns:
            Optional[tuple[int, int]]: (start, end) position of tool call tokens in response,
                                        or None if no tool call found.
        """
        if not agent_data.response_ids:
            return None
        
        segment = self._split_tool_call_segment(agent_data.response_ids)
        if not segment or not segment.get("call_ids"):
            return None
        
        # Calculate position in the full response (relative to response_mask)
        total_response_len = len(agent_data.response_mask)
        tool_call_len = len(segment["call_ids"])
        start = total_response_len - tool_call_len
        end = total_response_len
        
        return (start, end)
    
    def _mask_previous_code_blocks(
        self, prompt_ids: list[int], current_mask: list[int], current_response_ids: list[int]
    ) -> list[int]:
        """Find and mask ```python code blocks in previous assistant turns."""
        import re
        
        # Decode the full conversation except the current assistant turn
        current_turn_length = len(current_response_ids) if current_response_ids else 0
        previous_tokens = prompt_ids[: len(prompt_ids) - current_turn_length]
        
        if not previous_tokens:
            return current_mask
        
        # Decode to find code blocks
        previous_text = self.tokenizer.decode(previous_tokens, skip_special_tokens=False)
        
        # Find all ```python ... ``` blocks
        pattern = r'```python\s*(.*?)```'
        matches = list(re.finditer(pattern, previous_text, re.DOTALL))
        
        if not matches:
            return current_mask
        
        # For each code block, find its token span and mask it
        for match in matches:
            code_start_char = match.start()
            code_end_char = match.end()
            
            # Find the token indices corresponding to this character range
            text_before = previous_text[:code_start_char]
            text_with_code = previous_text[:code_end_char]
            
            tokens_before = self.tokenizer.encode(text_before, add_special_tokens=False)
            tokens_with_code = self.tokenizer.encode(text_with_code, add_special_tokens=False)
            
            # The code block tokens are the difference
            code_token_start = len(tokens_before)
            code_token_end = len(tokens_with_code)
            
            # Mask these tokens in current_mask
            if code_token_start < len(current_mask) and code_token_end <= len(current_mask):
                for i in range(code_token_start, code_token_end):
                    current_mask[i] = 1
        
        return current_mask
