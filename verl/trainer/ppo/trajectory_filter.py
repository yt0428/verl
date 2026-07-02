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
"""Trajectory filter: mask environment-corrupted trajectories out of the loss and group stats.

Agentic rollouts can fail for reasons unrelated to policy quality: sandbox/env setup
failure (dummy 1-token trajectories), agent or verifier wall-clock timeout, context-length
overflow, response-length truncation (DAPO "overlong"), turn-cap exhaustion. Their reward
(usually a spurious 0) reflects the environment, not the policy. With GRPO
std-normalization a spurious 0 both trains AGAINST the (possibly fine) trajectory and
inflates the advantages of its group siblings.

``apply_trajectory_filter`` neutralizes flagged trajectories without changing batch shapes:

1. ``token_level_rewards`` of flagged rows are zeroed (no reward signal can leak);
2. their ``uid`` is rewritten to a singleton group, removing them from the GRPO group
   statistics of their siblings (singleton groups get mean=0/std=1, so with a zeroed
   reward the resulting advantage is exactly 0);
3. a boolean row mask is stored in ``non_tensor_batch[FILTER_MASK_KEY]`` so
   ``zero_filtered_advantages`` can zero advantages/returns after any advantage
   estimator ran (belt-and-braces for non-GRPO estimators).

Flags come from the agent loop's ``extra_fields`` (propagated into ``non_tensor_batch``
by ``AgentLoopWorker._postprocess``), with trainer-side fallbacks computed from tensors.

Configured by ``algorithm.trajectory_filter`` (see TrajectoryFilterConfig); disabled by
default. Per-reason counts are logged under ``trajectory_filter/``.
"""

import numpy as np
import torch

from verl import DataProto

FILTER_MASK_KEY = "trajectory_filter_mask"


def _flag_array(batch: DataProto, key: str) -> np.ndarray:
    """Read a boolean per-row flag from non_tensor_batch; missing key/None entries -> False."""
    n = len(batch)
    if key not in batch.non_tensor_batch:
        return np.zeros(n, dtype=bool)
    return np.array([bool(x) for x in batch.non_tensor_batch[key]], dtype=bool)


def _response_token_counts(batch: DataProto) -> torch.Tensor:
    """Number of attended tokens in the response window, shape (bs,)."""
    response_len = batch.batch["responses"].shape[1]
    return batch.batch["attention_mask"][:, -response_len:].sum(dim=-1)


def apply_trajectory_filter(batch: DataProto, config) -> tuple[DataProto, dict]:
    """Flag env-corrupted trajectories and neutralize their reward/group contribution.

    Must run AFTER ``token_level_rewards`` is set and BEFORE advantage computation.

    Args:
        batch: training batch with ``responses``, ``attention_mask``,
            ``token_level_rewards`` and (for GRPO) ``non_tensor_batch["uid"]``.
        config: ``algorithm.trajectory_filter`` (TrajectoryFilterConfig / DictConfig / None).

    Returns:
        (batch, metrics): batch is modified in place; metrics carry per-reason counts.
    """
    if config is None or not config.get("enable", False):
        return batch, {}

    n = len(batch)
    invalid = np.zeros(n, dtype=bool)
    reasons: dict[str, np.ndarray] = {}

    if config.get("filter_env_failures", True):
        mask = _flag_array(batch, "is_env_failure")
        if "trajectory_token_source" in batch.non_tensor_batch:
            mask |= np.array(
                [x == "dummy" for x in batch.non_tensor_batch["trajectory_token_source"]], dtype=bool
            )
        min_tokens = int(config.get("min_response_tokens", 0))
        if min_tokens > 0:
            mask |= (_response_token_counts(batch) < min_tokens).cpu().numpy()
        reasons["env_failure"] = mask

    if config.get("filter_timeouts", True):
        reasons["timeout"] = _flag_array(batch, "is_timeout")

    if config.get("filter_context_errors", True):
        reasons["context_error"] = _flag_array(batch, "is_context_error")

    if config.get("filter_overlong", True):
        mask = _flag_array(batch, "is_overlong")
        # Fallback: a response that fills the whole window was truncated by the cap.
        response_len = batch.batch["responses"].shape[1]
        mask |= (_response_token_counts(batch) >= response_len).cpu().numpy()
        reasons["overlong"] = mask

    max_turns = int(config.get("max_num_turns", 0))
    if config.get("filter_turn_capped", False) and max_turns > 0 and "__num_turns__" in batch.non_tensor_batch:
        num_turns = np.asarray(batch.non_tensor_batch["__num_turns__"], dtype=np.int64)
        reasons["turn_capped"] = num_turns >= max_turns

    for mask in reasons.values():
        invalid |= mask

    metrics = {f"trajectory_filter/{name}_count": int(mask.sum()) for name, mask in reasons.items()}
    metrics["trajectory_filter/invalid_count"] = int(invalid.sum())
    metrics["trajectory_filter/invalid_ratio"] = float(invalid.mean()) if n else 0.0

    if invalid.any():
        invalid_t = torch.from_numpy(invalid).to(batch.batch["token_level_rewards"].device)
        # 1) no reward signal may leak through the singleton group
        batch.batch["token_level_rewards"][invalid_t] = 0.0
        # 2) remove from sibling GRPO group statistics via singleton uid
        if "uid" in batch.non_tensor_batch:
            uids = batch.non_tensor_batch["uid"]
            for i in np.nonzero(invalid)[0]:
                uids[i] = f"{uids[i]}__filtered_{i}"
        # 3) remember rows for post-advantage zeroing
        batch.non_tensor_batch[FILTER_MASK_KEY] = invalid

    return batch, metrics


def zero_filtered_advantages(batch: DataProto) -> DataProto:
    """Zero advantages/returns of rows flagged by ``apply_trajectory_filter``.

    Estimator-agnostic safety net: a zero advantage contributes exactly zero policy
    gradient regardless of clipping or IS weighting. No-op when nothing was flagged.
    """
    if FILTER_MASK_KEY not in batch.non_tensor_batch:
        return batch
    invalid = torch.from_numpy(np.asarray(batch.non_tensor_batch[FILTER_MASK_KEY], dtype=bool))
    for key in ("advantages", "returns"):
        if key in batch.batch:
            batch.batch[key][invalid.to(batch.batch[key].device)] = 0.0
    return batch
