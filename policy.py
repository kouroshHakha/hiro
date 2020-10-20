
from typing import cast, Tuple, Mapping

import abc
import torch
import torch.nn as nn
from torch.optim import Adam

from replay import ReplayTD3, BatchT
from network import ActorTD3, CriticTD3
from copy import deepcopy


class TD3Policy(abc.ABC):

    def __init__(self,
                 replay_buffer: ReplayTD3,
                 actor: ActorTD3,
                 critic: CriticTD3,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 polyak: float = 0.995,
                 bsize: int = 32,
                 policy_noise_std: float = 1,
                 policy_noise_clip: float = 1,
                 discount: float = 0.999,
                 device: torch.device = torch.device('cpu'),
    ):

        self._buffer = replay_buffer

        self._actor = actor.to(device)
        self._critic = critic.to(device)
        self._actor_target = deepcopy(actor)
        self._critic_target = deepcopy(critic)

        self.device = device
        self.bsize = bsize

        self.policy_noise_std = policy_noise_std
        self.policy_noise_clip = policy_noise_clip
        self.action_limit = self._actor.action_limit
        self.discount = discount
        self.polyak = polyak


        self.q_params = self._critic.parameters()
        self.critic_optim = Adam(self.q_params, lr=critic_lr)
        self.actor_optim = Adam(self._actor.parameters(), lr=actor_lr)

    @property
    def buffer(self) -> ReplayTD3:
        return self._buffer

    @property
    def actor(self) -> ActorTD3:
        return self._actor

    def step_update(self, batch: BatchT, update_actor_and_target_nn: bool = False):
        loss_q, loss_q_info = self._compute_loss_q(batch)
        self.critic_optim.zero_grad()
        loss_q.backward()
        self.critic_optim.step()

        if update_actor_and_target_nn:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            loss_pi = self._compute_loss_pi(batch)
            self.actor_optim.zero_grad()
            loss_pi.backward()
            self.actor_optim.step()

            # update actor and critic target networks with polyak averaging
            with torch.no_grad():
                # actor
                for p, p_targ in zip(self._actor.parameters(), self._actor_target.parameters()):
                    p_targ.data = self.polyak * p_targ + (1 - self.polyak) * p.data
                # critic
                for p, p_targ in zip(self._critic.parameters(), self._critic_target.parameters()):
                    p_targ.data = self.polyak * p_targ + (1 - self.polyak) * p.data

    def _compute_loss_q(self, batch: BatchT) -> Tuple[torch.Tensor, Mapping[str, float]]:
        state, action, reward, next_state, done = self._parse_batchT(batch)

        q1, q2 = self._critic(state, action)

        # Bellman backup for Q functions
        with torch.no_grad:
            next_action = self._actor_target(next_state)

            noise = torch.randn(self._actor.action_dim) * self.policy_noise_std
            eps = torch.clamp(noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_action = torch.clamp(next_action + eps, -self.action_limit, self.action_limit)

            # Target Q-values
            q1_targ, q2_targ = self._critic_target(next_state, next_action, return_both=True)
            q_targ = torch.min(q1_targ, q2_targ)
            backup = reward + self.discount * (1 - done) * q_targ

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        loss_info = dict(q1_loss=loss_q1.item(), q2_loss=loss_q2.item())

        return loss_q, loss_info

    def _compute_loss_pi(self, batch: BatchT):
        state, _, _, _, _ = self._parse_batchT(batch)
        q1 = self._critic(state, self._actor(state), return_both=False)
        return -q1.mean()

    def _parse_batchT(self, batch: BatchT) -> Tuple[torch.Tensor, ...]:
        state = batch['state']
        next_state = batch['next_state']
        action = batch['action']
        rew = batch['rew']
        done = batch['done']
        return state, action, rew, next_state, done

