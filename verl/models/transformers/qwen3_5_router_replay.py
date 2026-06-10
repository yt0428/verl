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
"""FSDP / HuggingFace MoE router replay for Qwen3.5-MoE.

Upstream ``verl/utils/megatron/router_replay_patch.py`` patches Megatron's
``TopKRouter``; it cannot serve the FSDP path, which trains the HF transformers
model directly. This module is the HF analogue: it patches
``Qwen3_5MoeTopKRouter.forward`` and drives it through the SAME backend-agnostic
``RouterReplay`` registry (``verl.utils.router_replay``) the Megatron path uses,
so both share one framework and one ``router_replay.mode`` config.

Difference vs the Megatron router (R2/R3): rollout-stage R3 only records routing
for the assistant tokens vLLM actually generated; prompt / tool / injected tokens
carry an all-zero placeholder row. A real top-k over 256 experts never yields all
zeros, so we replay ONLY non-placeholder rows and keep the model's own routing for
placeholder rows. (Megatron's R2 records every token, so it has no placeholders.)

The replay GATING WEIGHTS are recomputed from the current ``router_logits`` via
``gather`` over the replayed indices, so gradients still flow into the gate and the
router keeps training; only the expert *selection* is forced to rollout's choice.
"""

import logging
import os

import torch

from verl.utils.router_replay import RouterReplay, RouterReplayAction

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_R3_REPLAY_LOGGED = 0


def _make_r3_router_forward(orig_forward):
    """Build the replay-aware ``Qwen3_5MoeTopKRouter.forward`` wrapping ``orig_forward``.

    When the router's attached ``RouterReplay`` instance has no action set (non-R3
    runs, ref policy, eval), this delegates to ``orig_forward`` verbatim — byte
    identical, zero overhead. Otherwise it reimplements the router (softmax → top-k
    → renormalize) with the top-k selection overridden for REPLAY_FORWARD or the
    indices recorded for RECORD.

    NOTE (human review): the replay-path reimplementation must match the installed
    transformers ``Qwen3_5MoeTopKRouter.forward`` (softmax dtype, return tuple
    order). Verify against the transformers version pinned for this env.
    """

    def _r3_router_forward(self, hidden_states):
        rr = getattr(self, "_router_replay", None)
        action = rr.router_replay_action if rr is not None else None
        if action is None:
            # Disabled path: exactly the original router.
            return orig_forward(self, hidden_states)

        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        n_tokens = router_indices.shape[0]

        if action == RouterReplayAction.RECORD:
            rr.record_indices(router_indices)
        elif action == RouterReplayAction.REPLAY_FORWARD and rr.target_topk_idx is not None:
            target = rr.target_topk_idx
            if target.shape[0] == n_tokens:
                replay_idx = target.to(router_indices.device).long()  # [n, k]
                real = (replay_idx != 0).any(dim=-1, keepdim=True)  # placeholder rows -> False
                router_indices = torch.where(real, replay_idx, router_indices)
                router_top_value = router_probs.gather(-1, router_indices)
                global _R3_REPLAY_LOGGED
                if _R3_REPLAY_LOGGED < 5:  # proof the training-side replay actually executed
                    _R3_REPLAY_LOGGED += 1
                    logger.warning(
                        "[R3-replay] training-side replay FIRED: tokens=%d replayed_rows=%d",
                        n_tokens,
                        int(real.sum()),
                    )
            else:
                logger.warning(
                    "R3 replay token count %d != router tokens %d; skipping replay this forward "
                    "(check remove-padding alignment / sp_size==1).",
                    target.shape[0],
                    n_tokens,
                )

        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    return _r3_router_forward


def apply_qwen3_5_router_replay_patch(model) -> int:
    """Install R3 router replay on a Qwen3.5-MoE model for the FSDP path.

    1. Idempotently replace ``Qwen3_5MoeTopKRouter.forward`` (class-level) with the
       replay-aware version; the disabled path keeps the original verbatim.
    2. Attach a fresh ``RouterReplay`` instance to each gate, in decoder-layer
       (module enumeration) order, so ``RouterReplay.set_replay_data([per-layer
       indices])`` distributes layer L's indices to layer L's gate. All 40
       Qwen3.5-MoE layers are MoE, so enumeration order == layer id == the vLLM
       capturer's layer axis.

    Returns the number of MoE gates patched (== num MoE layers).
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter

    if not getattr(Qwen3_5MoeTopKRouter, "_r3_patched", False):
        Qwen3_5MoeTopKRouter.forward = _make_r3_router_forward(Qwen3_5MoeTopKRouter.forward)
        Qwen3_5MoeTopKRouter._r3_patched = True

    # Reset the global registry so re-applying (e.g. a rebuilt model) doesn't leave
    # stale instances behind, which would make set_replay_data's length check fail.
    # Safe on the FSDP path: this process never registers Megatron routers (their
    # module hard-imports megatron and isn't loaded here).
    RouterReplay.router_instances = []
    n = 0
    for module in model.modules():
        if module.__class__.__name__ == "Qwen3_5MoeTopKRouter":
            module._router_replay = RouterReplay()  # __init__ appends to the registry, in layer order
            n += 1
    return n
