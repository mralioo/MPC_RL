# Copyright (C) 2018, Anass Al, Juan Camilo Gamboa Higuera
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Double cartpole environment."""

import torch
import numpy as np

from gym import spaces

import torch
import numpy as np

from gym import spaces

from RL.utils import *


class DoubleCartpoleReward(torch.nn.Module):
    def __init__(self,
                 pole1_length=torch.tensor(0.5),
                 pole2_length=torch.tensor(0.5),
                 target=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 Q=8.0 * torch.eye(2),
                 R=1e-3 * torch.eye(1)):
        super(DoubleCartpoleReward, self).__init__()
        self.Q = torch.nn.Parameter(Q, requires_grad=False)
        self.R = torch.nn.Parameter(R, requires_grad=False)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        self.target = torch.nn.Parameter(target, requires_grad=False)
        self.pole1_length = torch.nn.Parameter(pole1_length,
                                               requires_grad=False)
        self.pole2_length = torch.nn.Parameter(pole2_length,
                                               requires_grad=False)

    def forward(self, x, u):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u)
        x = x.to(device=self.Q.device, dtype=self.Q.dtype)
        u = u.to(device=self.Q.device, dtype=self.Q.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        # compute the distance between the tip of the pole and the target tip
        # location
        targeta = to_complex(self.target, [2, 4])  # [others, np.sin(angles), np.cos(angles)]
        target_tip_xy = torch.cat([
            targeta[:, 0:1] - self.pole1_length * targeta[:, 4:5] -
            self.pole2_length * targeta[:, 5:6],
            self.pole1_length * targeta[:, 6:7] +
            self.pole2_length * targeta[:, 7:8]
        ],
            dim=-1)

        if x.shape[-1] != targeta.shape[-1]:
            xa = to_complex(x, [2, 4])
        else:
            xa = x
        pole_tip_xy = torch.cat([
            xa[:, 0:1] - self.pole1_length * xa[:, 4:5] -
            self.pole2_length * xa[:, 5:6],
            self.pole1_length * xa[:, 6:7] + self.pole2_length * xa[:, 7:8]
        ],
            dim=-1)

        pole_tip_xy = pole_tip_xy.unsqueeze(
            0) if pole_tip_xy.dim() == 1 else pole_tip_xy
        target_tip_xy = target_tip_xy.unsqueeze(
            0) if target_tip_xy.dim() == 1 else target_tip_xy

        delta = pole_tip_xy - target_tip_xy
        delta = delta / (2 * (self.pole1_length + self.pole2_length))
        cost = 0.5 * ((delta.mm(self.Q) * delta).sum(-1, keepdim=True) +
                      (u.mm(self.R) * u).sum(-1, keepdim=True))
        # reward is negative cost.
        # optimizing the exponential of the negative cost
        reward = (-cost).exp()
        return reward


def reward_func(state, action):
    # reward for maintaining the pole upright
    angle_rw = (torch.cos(state[..., 1:2]) + 1) / 2.0

    # reward for maintaining the cart centered
    scale = np.sqrt(-2 * np.log(0.1)) / 2.0
    cart_rw = (1 + torch.exp(-0.5 * (state[..., 0:1] * scale) ** 2)) / 2

    # control penalty (reduces reward for larger actions)
    control_rw = (4 +
                  torch.max(torch.zeros_like(action), 1 - action ** 2)) / 5

    # velocity penalty (halves the reward if spinning too fast)
    vel_rw = (1 + torch.exp(-0.5 * state[..., 3:4] ** 2)) / 2

    return angle_rw * cart_rw * control_rw * vel_rw
