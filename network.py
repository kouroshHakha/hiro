"""
Networks
"""
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class ActorTD3(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 action_scale: Union[np.ndarray, torch.Tensor]):
        super(ActorTD3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )

        if isinstance(action_scale, np.ndarray):
            self.output_scale = torch.from_numpy(action_scale)
        else:
            self.output_scale = action_scale

    def forward(self, state: torch.Tensor):
        if state.ndim == 1:
            state = state[None, :]
        logits = self.fc(state)
        action = logits * self.output_scale
        return action


class CriticTD3(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(CriticTD3, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, return_both: bool = True):
        if state.ndim == 1:
            state = state[None, :]
        if action.ndim == 1:
            action = action[None, :]

        state_action = torch.cat([state, action], -1)
        q1 = self.q1(state_action)
        if return_both:
            q2 = self.q2(state_action)
            return q1, q2

        return q1


class ActorLow(ActorTD3):
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, action_scale: float):
        scale = torch.tensor([action_scale])
        super().__init__(state_dim=state_dim + goal_dim, action_dim=action_dim, action_scale=scale)


class ActorHigh(ActorTD3):
    def __init__(self, state_dim: int, goal_dim: int, goal_scale: Union[np.ndarray, torch.Tensor]):
        super().__init__(state_dim=state_dim, action_dim=goal_dim, action_scale=goal_scale)


class CriticLow(CriticTD3):
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim+goal_dim, action_dim=action_dim)

class CriticHi(CriticTD3):
    def __init__(self, state_dim: int, goal_dim: int):
        super().__init__(state_dim=state_dim, action_dim=goal_dim)