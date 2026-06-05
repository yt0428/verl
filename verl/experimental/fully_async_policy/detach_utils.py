# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    rollout_status: dict[str, Any]


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    existing_batch_keys = [k for k in batch_keys_to_pop if k in full_batch.batch.keys()]
    existing_non_tensor_keys = [k for k in non_tensor_batch_keys_to_pop if k in full_batch.non_tensor_batch.keys()]

    if existing_batch_keys or existing_non_tensor_keys:
        full_batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=existing_non_tensor_keys,
        )

    # Setting selected agent, that supports partial
    if config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(["tool_agent"] * len(full_batch), dtype=object)
    else:
        # full_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(full_batch), dtype=object)
        pass

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    return full_batch


# Same semantic keys as ``AgentLoopManager._performance_metrics`` (agent_loop.py).
_AGENT_LOOP_STANDARD_METRIC_KEYS = frozenset(
    {"generate_sequences", "tool_calls", "compute_score", "num_preempted"}
)


def _aggregate_agent_loop_timing_meta(final_batch: DataProto) -> dict[str, float]:
    """Mirror :meth:`AgentLoopManager._performance_metrics` for fully-async assembly.

    Populates ``timing_s/agent_loop/...`` keys so ``FullyAsyncTrainer._collect_metrics_from_samples``
    forwards them into ``self.metrics`` and loggers (same surface as synchronous rollout).
    """
    pt = np.asarray(final_batch.non_tensor_batch["processing_times"], dtype=np.float64)
    tc = np.asarray(final_batch.non_tensor_batch["tool_calls_times"], dtype=np.float64)
    n = pt.shape[0]
    if "compute_score_times" in final_batch.non_tensor_batch:
        t_cs = np.asarray(final_batch.non_tensor_batch["compute_score_times"], dtype=np.float64)
    else:
        t_cs = np.zeros(n, dtype=np.float64)
    if "num_preempted_vals" in final_batch.non_tensor_batch:
        npm = np.asarray(final_batch.non_tensor_batch["num_preempted_vals"], dtype=np.float64)
    else:
        npm = np.full(n, -1.0, dtype=np.float64)

    slowest = int(np.argmax(pt + tc + t_cs)) if n > 0 else 0

    out: dict[str, float] = {}
    for name, arr in (
        ("generate_sequences", pt),
        ("tool_calls", tc),
        ("compute_score", t_cs),
        ("num_preempted", npm),
    ):
        if arr.size == 0:
            continue
        out[f"timing_s/agent_loop/{name}/min"] = float(arr.min())
        out[f"timing_s/agent_loop/{name}/max"] = float(arr.max())
        out[f"timing_s/agent_loop/{name}/mean"] = float(arr.mean())
        if slowest < arr.size:
            out[f"timing_s/agent_loop/slowest/{name}"] = float(arr[slowest])

    extra_keys: set[str] = set()
    for nk in final_batch.non_tensor_batch.keys():
        if nk.startswith("_al_timing_"):
            extra_keys.add(nk[len("_al_timing_") :])
    for key in sorted(extra_keys):
        vals = np.asarray(final_batch.non_tensor_batch[f"_al_timing_{key}"], dtype=np.float64)
        if vals.size == 0:
            continue
        out[f"timing_s/agent_loop/{key}/min"] = float(vals.min())
        out[f"timing_s/agent_loop/{key}/max"] = float(vals.max())
        out[f"timing_s/agent_loop/{key}/mean"] = float(vals.mean())
        if slowest < vals.size:
            out[f"timing_s/agent_loop/slowest/{key}"] = float(vals[slowest])

    if (
        n > 0
        and "attention_mask" in final_batch.batch
        and "prompts" in final_batch.batch
        and slowest < n
    ):
        prompt_length = int(final_batch.batch["prompts"].shape[1])
        attention_mask = final_batch.batch["attention_mask"][slowest]
        plen = min(prompt_length, int(attention_mask.shape[0]))
        out["timing_s/agent_loop/slowest/prompt_length"] = float(attention_mask[:plen].sum().item())
        out["timing_s/agent_loop/slowest/response_length"] = float(attention_mask[plen:].sum().item())

    return out


def addition_process(output: DataProto):
    """collect metirics"""
    metrics_list = output.meta_info.pop("metrics")  # List[Dict[str, Any]]
    output.non_tensor_batch["processing_times"] = np.asarray(
        [item["generate_sequences"] for item in metrics_list], dtype=np.float64
    )
    output.non_tensor_batch["tool_calls_times"] = np.asarray(
        [item["tool_calls"] for item in metrics_list], dtype=np.float64
    )
    output.non_tensor_batch["compute_score_times"] = np.asarray(
        [float(item.get("compute_score") or 0.0) for item in metrics_list], dtype=np.float64
    )
    output.non_tensor_batch["num_preempted_vals"] = np.asarray(
        [float(item.get("num_preempted", -1)) for item in metrics_list], dtype=np.float64
    )

    extra_keys: set[str] = set()
    for m in metrics_list:
        extra_keys.update(m.keys())
    extra_keys -= _AGENT_LOOP_STANDARD_METRIC_KEYS
    for key in sorted(extra_keys):
        output.non_tensor_batch[f"_al_timing_{key}"] = np.asarray(
            [float(m.get(key) or 0.0) for m in metrics_list], dtype=np.float64
        )

    return output


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        batch = addition_process(rs.full_batch)
        rollout_samples_batch.append(batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    # Align with ``AgentLoopManager._performance_metrics`: Harbor / extra agent-loop timings → log.
    agent_loop_timing_meta = _aggregate_agent_loop_timing_meta(final_batch)
    for nk in list(final_batch.non_tensor_batch.keys()):
        if nk.startswith("_al_timing_"):
            final_batch.non_tensor_batch.pop(nk, None)
    for nk in ("compute_score_times", "num_preempted_vals"):
        final_batch.non_tensor_batch.pop(nk, None)

    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **agent_loop_timing_meta,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t:.4f} seconds.")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["fully_async/trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }


def task_exception_handler(task: asyncio.Task):
    """Handle task exceptions and log them"""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is expected
    except Exception as e:
        print(f"Task {task.get_name()} failed with exception: {e}")
        raise e


def safe_create_task(coro, name: str, task_set: set = None):
    """Safely create a task with exception handling

    Args:
        coro: The coroutine to run
        name: Name for the task
        task_set: Optional set to add the task to

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(task_exception_handler)
    if task_set is not None:
        task_set.add(task)
    return task
