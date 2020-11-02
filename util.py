from typing import Union

import numpy as np
import torch
from torch import Tensor

from environments.create_maze_env import create_maze_env


ArrayT =  Union[np.ndarray, Tensor]

def get_goal_scale(env_name: str, use_torch: bool = True,
                   device: torch.device = torch.device('cpu')) -> ArrayT:
    arr = np.array([10., 10., 0.5])
    if use_torch:
        return torch.tensor(arr, dtype=torch.float32).to(device)
    return arr

def get_target_position(env_name: str, use_torch: bool = True,
                        device: torch.device = torch.device('cpu')) -> ArrayT:
    if env_name == 'AntPush':
        target_pos = np.array([0, 19, 0.5])
    elif env_name == 'AntFall':
        target_pos = np.array([0, 27, 4.5])
    else:
        raise ValueError(f"{env_name} is either wrong or not implemented!")

    if use_torch:
        return torch.from_numpy(target_pos).to(device)
    return target_pos

def h_function(state: ArrayT, goal: ArrayT, next_state: ArrayT, goal_dim: int) -> ArrayT:
    # return next goal
    return state[:goal_dim] + goal - next_state[:goal_dim]

def intrinsic_reward(state: ArrayT, goal: ArrayT, next_state: ArrayT, goal_dim: int) -> ArrayT:
    # low-level dense reward (L2 norm), provided by high-level policy
    return -(((state[:goal_dim] + goal - next_state[:goal_dim]) ** 2).sum(-1)) ** 0.5

def dense_reward(state: ArrayT, target: ArrayT, goal_dim) -> ArrayT:
    if type(state) != type(target):
        raise ValueError(f'state has incompatible type with target {type(state)} != {type(target)}')

    if torch.is_tensor(target):
        target = target.to(state)

    return - (((state[:goal_dim] - target) ** 2).sum(-1)) ** 0.5

def done_judge_low(goal: ArrayT) -> ArrayT:
    # define low-level success: same as high-level success (L2 norm < 5, paper B.2.2)
    l2_norm = ((goal ** 2).sum(-1)) ** 0.5
    # Note: not clear in the paper (should look at the code) but code base uses this criteria to
    # induce early variation in done_l signal
    return l2_norm <= 1.5

def success_judge(state: ArrayT, target: ArrayT, goal_dim: int) -> ArrayT:
    l2_norm = (((state[:goal_dim] - target) ** 2).sum(-1)) ** 0.5
    return l2_norm <= 5.
