from typing import Union, Dict

import abc
import numpy as np
import torch

BatchT = Dict[str, Union[np.ndarray, torch.Tensor]]


class ReplayTD3(abc.ABC):

    @abc.abstractmethod
    def store(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def sample_batch(self, batch_size: int = 32, return_tensor: bool = False,
                     device: torch.device = torch.device('cpu')) -> BatchT:
        pass


class ReplayBufferLo(ReplayTD3):

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, size: int = 200000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.goal_buf = np.zeros((size, goal_dim), dtype=np.float32)
        self.next_goal_buf = np.zeros((size, goal_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs: np.ndarray, goal: np.ndarray, act: np.ndarray,
              rew: float, next_obs: np.ndarray, next_goal: np.ndarray, done: float):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.goal_buf[self.ptr] = goal
        self.next_goal_buf[self.ptr] = next_goal
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size: int = 32, return_tensor: bool = False,
                     device: torch.device = torch.device('cpu')) -> BatchT:
        idxs = np.random.randint(0, self.size, size=batch_size)

        state = np.concatenate([self.obs_buf[idxs], self.goal_buf[idxs]], axis=-1)
        next_state = np.concatenate([self.next_obs_buf[idxs], self.next_goal_buf[idxs]], axis=-1)

        batch = dict(
            state=state,
            next_state=next_state,
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs]
        )

        if return_tensor:
            return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}
        else:
            return batch

class ReplayBufferHi(ReplayTD3):

    def __init__(self, obs_dim: int, goal_dim: int, size: int = 200000):
        self.obs_start_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs_end_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.goal_start_buf = np.zeros((size, goal_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs_start: np.ndarray, goal_start: np.ndarray, rew: float, obs_end: np.ndarray,
              done: float):
        self.obs_start_buf[self.ptr] = obs_start
        self.obs_end_buf[self.ptr] = obs_end
        self.goal_start_buf[self.ptr] = goal_start
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size: int = 32, return_tensor: bool = False,
                     device: torch.device = torch.device('cpu')) -> BatchT:
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            state=self.obs_start_buf[idxs],
            next_state=self.obs_end_buf[idxs],
            act=self.goal_start_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs]
        )

        if return_tensor:
            return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}
        else:
            return batch