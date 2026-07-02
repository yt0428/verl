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

"""Tier-1 unit test for the FSDP-path R3 router replay mechanism.

Validates ``verl/models/transformers/qwen3_5_router_replay.py`` in isolation —
no FSDP wrapping, no distributed init, no full training stack. Builds real
``transformers`` ``Qwen3_5MoeTopKRouter`` instances on a tiny config, installs
the replay patch, and drives the ``RouterReplay`` registry directly.

Covers the four properties the handoff
(``docs/training_env/r3-fsdp-on-v0.8.0-port-handoff-20260610.md`` §3 Tier 1)
calls out, plus open-item #2 (the reimplementation must match the *installed*
``transformers`` router byte-for-byte on non-replayed rows):

1. REPLAY_FORWARD forces the top-k *selection* of real (non-placeholder) rows to
   the injected rollout indices.
2. All-zero placeholder rows keep the model's own routing.
3. Gating weights still carry a gradient back to ``router.weight``.
4. (open-item #2) With replay armed but every row a placeholder, the patched
   router output is identical to the stock ``Qwen3_5MoeTopKRouter.forward`` —
   same dtype, same values, same return-tuple order.

Runs on CPU; opportunistically uses CUDA (bf16) when a GPU is visible, which is
the production path. Standalone runner at the bottom so it works with or without
pytest: ``python tests/models/test_qwen3_5_router_replay_on_cpu.py``.
"""

from types import SimpleNamespace

import torch

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter

from verl.models.transformers.qwen3_5_router_replay import apply_qwen3_5_router_replay_patch
from verl.utils.router_replay import RouterReplay, RouterReplayAction

# Capture the pristine forward BEFORE any patch is installed (the patch is
# class-level and permanent), so the equivalence tests can compute a stock
# reference even after the class has been patched by an earlier test.
_ORIG_FORWARD = Qwen3_5MoeTopKRouter.forward

# Tiny but representative: top_k > 1 (so a real row's k distinct experts can
# never be all-zero, which is what distinguishes it from a placeholder row),
# num_experts large enough that expert 0 is not special.
N_EXPERTS = 32
TOP_K = 4
HIDDEN = 16
N_TOKENS = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def _cfg():
    return SimpleNamespace(num_experts_per_tok=TOP_K, num_experts=N_EXPERTS, hidden_size=HIDDEN)


class _GateStack(torch.nn.Module):
    """A bare container of N ``Qwen3_5MoeTopKRouter`` gates, standing in for the
    decoder-layer enumeration order that ``apply_qwen3_5_router_replay_patch``
    walks via ``model.modules()``."""

    def __init__(self, n_layers, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        gates = []
        for _ in range(n_layers):
            g = Qwen3_5MoeTopKRouter(_cfg())
            # __init__ zero-inits the gate weight (-> uniform routing, degenerate
            # ties). Give each gate distinct non-zero weights for meaningful top-k.
            g.weight.data = torch.randn(N_EXPERTS, HIDDEN, generator=gen)
            gates.append(g)
        self.gates = torch.nn.ModuleList(gates)


def _reset_registry():
    """Drop any router instances / actions left by a previous test. The patch
    itself resets ``router_instances`` on each apply, but clear actions too so a
    stale REPLAY_FORWARD can never leak across tests."""
    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_indices()
    RouterReplay.router_instances = []


def _make_one_gate(seed=0):
    _reset_registry()
    stack = _GateStack(1, seed=seed).to(DEVICE, DTYPE)
    n = apply_qwen3_5_router_replay_patch(stack)
    assert n == 1, f"expected 1 gate patched, got {n}"
    return stack.gates[0]


def _input(n_tokens=N_TOKENS, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n_tokens, HIDDEN, generator=gen).to(DEVICE, DTYPE)


# ============================================================ open-item #2
def test_reimpl_matches_stock_on_placeholder_rows():
    """Replay armed, but EVERY row is an all-zero placeholder -> no selection is
    overridden, yet the full reimplementation path (softmax -> top-k -> gather ->
    renorm -> dtype) executes. Its output must equal the stock router exactly:
    same first-return (float32 softmax probs), same float32 gating weights, same
    indices. This is the regression guard for the dtype / return-tuple port bug."""
    gate = _make_one_gate()
    x = _input()

    # Stock reference via the pristine (unpatched) forward on the same weights.
    stock_logits, stock_scores, stock_idx = _ORIG_FORWARD(gate, x)

    RouterReplay.set_replay_data([torch.zeros(N_TOKENS, TOP_K, dtype=torch.long, device=DEVICE)])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    re_logits, re_scores, re_idx = gate(x)

    assert re_logits.dtype == stock_logits.dtype, (
        f"first-return dtype {re_logits.dtype} != stock {stock_logits.dtype} "
        f"(stock returns the float32 softmax probs, not raw logits)"
    )
    assert re_scores.dtype == stock_scores.dtype, (
        f"gating-weight dtype {re_scores.dtype} != stock {stock_scores.dtype} "
        f"(stock casts to the softmax/float32 dtype)"
    )
    assert torch.equal(re_idx, stock_idx), "placeholder rows must keep the model's own top-k selection"
    assert torch.allclose(re_scores.float(), stock_scores.float(), atol=1e-6, rtol=0), "gating weights diverged"
    assert torch.allclose(re_logits.float(), stock_logits.float(), atol=1e-6, rtol=0), "router probs diverged"


# ============================================================ properties 1 & 2
def test_replay_overrides_real_rows_keeps_placeholders():
    """Real (non-zero) rows replay the injected indices verbatim; all-zero rows
    keep the model's own top-k. Gating weights renormalize to 1 on every row."""
    gate = _make_one_gate()
    x = _input()
    _, _, stock_idx = _ORIG_FORWARD(gate, x)

    target = torch.zeros(N_TOKENS, TOP_K, dtype=torch.long, device=DEVICE)
    real_rows = [0, 2, 4]
    placeholder_rows = [1, 3, 5]
    target[0] = torch.tensor([3, 7, 15, 29], device=DEVICE)
    target[2] = torch.tensor([1, 9, 17, 31], device=DEVICE)
    target[4] = torch.tensor([2, 8, 16, 30], device=DEVICE)

    RouterReplay.set_replay_data([target])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    _, scores, idx = gate(x)

    for r in real_rows:
        assert torch.equal(idx[r], target[r]), f"real row {r} did not replay injected indices: {idx[r].tolist()}"
    for r in placeholder_rows:
        assert torch.equal(idx[r], stock_idx[r]), f"placeholder row {r} should keep model routing"
    sums = scores.float().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"gating weights not renormalized: {sums.tolist()}"


# ============================================================ property 3
def test_gating_weights_have_gradient():
    """Selection is forced, but gates are recomputed from current logits via
    gather -> gradient must still flow into router.weight (router keeps training)."""
    gate = _make_one_gate()
    gate.weight.requires_grad_(True)
    x = _input()

    target = torch.zeros(N_TOKENS, TOP_K, dtype=torch.long, device=DEVICE)
    target[0] = torch.tensor([3, 7, 15, 29], device=DEVICE)  # one real row, rest placeholders
    RouterReplay.set_replay_data([target])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    _, scores, _ = gate(x)
    scores.float().sum().backward()

    assert gate.weight.grad is not None, "no gradient reached router.weight"
    assert torch.isfinite(gate.weight.grad).all(), "router.weight grad has non-finite entries"
    assert gate.weight.grad.abs().sum().item() > 0.0, "router.weight grad is all-zero"


# ============================================================ RECORD path
def test_record_captures_model_indices_without_override():
    """RECORD stores the model's own top-k and does not alter the selection."""
    gate = _make_one_gate()
    x = _input()
    _, _, stock_idx = _ORIG_FORWARD(gate, x)

    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    _, _, idx = gate(x)

    assert torch.equal(idx, stock_idx), "RECORD must not override the selection"
    assert torch.equal(gate._router_replay.recorded_topk_idx, stock_idx), "RECORD did not capture model indices"


# ============================================================ disabled path
def test_disabled_path_is_verbatim():
    """With no action set, the patched forward delegates to the original — the
    non-R3 / ref-policy / eval path must be byte-identical and overhead-free."""
    gate = _make_one_gate()
    x = _input()
    stock = _ORIG_FORWARD(gate, x)
    # _router_replay attached but action is None.
    out = gate(x)
    for a, b in zip(out, stock):
        assert a.dtype == b.dtype and torch.equal(a, b), "disabled path diverged from stock router"


# ============================================================ layer ordering
def test_set_replay_data_distributes_per_layer_in_module_order():
    """``set_replay_data([t0, t1, t2])`` must hand layer i's indices to gate i,
    in ``model.modules()`` enumeration order (== decoder layer id == vLLM
    capturer layer axis)."""
    _reset_registry()
    stack = _GateStack(3, seed=2).to(DEVICE, DTYPE)
    n = apply_qwen3_5_router_replay_patch(stack)
    assert n == 3
    x = _input()

    # All-real (non-zero) per-layer targets so every row is fully overridden.
    base = torch.arange(1, N_TOKENS * TOP_K + 1, dtype=torch.long).reshape(N_TOKENS, TOP_K)
    targets = [((base + 3 * i) % (N_EXPERTS - 1) + 1).to(DEVICE) for i in range(3)]

    RouterReplay.set_replay_data(targets)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    for i, gate in enumerate(stack.gates):
        _, _, idx = gate(x)
        assert torch.equal(idx, targets[i]), f"gate {i} replayed the wrong layer's indices"


_TESTS = [
    test_reimpl_matches_stock_on_placeholder_rows,
    test_replay_overrides_real_rows_keeps_placeholders,
    test_gating_weights_have_gradient,
    test_record_captures_model_indices_without_override,
    test_disabled_path_is_verbatim,
    test_set_replay_data_distributes_per_layer_in_module_order,
]


if __name__ == "__main__":
    import traceback

    print(f"[router-replay tier1] device={DEVICE} dtype={DTYPE} "
          f"transformers Qwen3_5MoeTopKRouter @ {Qwen3_5MoeTopKRouter.__module__}")
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failures += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"[router-replay tier1] {len(_TESTS) - failures}/{len(_TESTS)} passed")
    raise SystemExit(1 if failures else 0)
