# Copyright 2026 The verl-agent contributors
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

import torch
from torch import nn


class LatentTransitionPredictor(nn.Module):
    """Small residual MLP for action-conditioned latent transition prediction."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int | None = None,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        bottleneck_size = bottleneck_size or hidden_size
        self.residual = residual
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_size, hidden_size),
        )
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        prediction = self.net(hidden_states)
        if self.residual:
            prediction = hidden_states + prediction
        return prediction
