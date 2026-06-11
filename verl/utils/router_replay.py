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
"""Backend-agnostic MoE router-replay holder.

The ``RouterReplay`` registry + ``RouterReplayAction`` enum are pure data
structures with no backend (Megatron / FSDP / VeOmni) dependency. They used to
live in ``verl/utils/megatron/router_replay_patch.py``, whose module-level
``import megatron`` makes it unimportable in a Megatron-free (e.g. FSDP) env.
Moved here so the FSDP engine can drive router replay through the same framework.
``router_replay_patch`` re-exports these names for backward compatibility.

Reference: https://github.com/THUDM/slime/blob/main/slime/utils/routing_replay.py
"""

from enum import Enum

import torch


class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    router_instances = []

    @staticmethod
    def set_replay_data(all_layers_topk_indices: list):
        """
        Distributes the topk indices for all layers to their respective RouterReplay instances.
        :param all_layers_topk_indices: A list of tensors, where each tensor contains the
                                        topk indices for a specific layer. The order
                                        must match the instantiation order of the routers.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.router_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match the number of router instances ({len(RouterReplay.router_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.router_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> list:
        """
        Collects the recorded topk indices from all RouterReplay instances.
        :return: A list of tensors, each containing the recorded topk indices for a layer.
        """
        return [router.get_recorded_indices() for router in RouterReplay.router_instances]

    @staticmethod
    def clear_global_indices():
        """Clears the recorded and target topk indices in all instances."""
        for router in RouterReplay.router_instances:
            router.clear_indices()

    def __init__(self):
        """Initializes a RouterReplay instance for a specific layer."""
        self.target_topk_idx = None  # For replay
        self.recorded_topk_idx = None  # For recording
        self.router_replay_action = None  # Router replay action for this layer
        self.replay_backward_list = []  # List of tensors for backward pass replay
        RouterReplay.router_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor):
        """Sets the target topk indices for replay."""
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self):
        """Returns the recorded topk indices."""
        return self.recorded_topk_idx

    def record_indices(self, topk_indices: torch.Tensor):
        """Records the topk indices."""
        self.recorded_topk_idx = topk_indices

    def clear_indices(self):
        """Clears the recorded and target topk indices."""
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        """Sets the router replay action for this layer."""
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        """Clears the router replay action for this layer."""
        self.router_replay_action = None

    @staticmethod
    def set_global_router_replay_action(router_replay_action: RouterReplayAction):
        """Sets the router replay action for all router instances."""
        for router in RouterReplay.router_instances:
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action():
        """Clears the router replay action for all router instances."""
        for router in RouterReplay.router_instances:
            router.clear_router_replay_action()
