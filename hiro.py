from typing import List


import torch
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from policy import TD3Policy
from network import ActorLow, CriticLow, ActorHigh, CriticHi
from replay import ReplayBufferHi, ReplayBufferLo
from environments.create_maze_env import create_maze_env
from util import (
    get_goal_scale, get_target_position, success_judge, h_function, intrinsic_reward,
    dense_reward, done_judge_low
)

@dataclass_json
@dataclass
class HiroConfig:
    env_name: str = 'AntPush'
    seed: int = 0

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    polyak: float = 0.995
    discount: float = 0.999
    bsize: int = 32
    epsiod_len: int = 10000
    replay_buffer_size: int = 200000
    max_timestep: int = 10e6
    hi_level_train_step: int = 10
    policy_noise_std: float = 1.0
    noise_clip: float = 1.0
    expl_noise_std_lo: float = 1.0
    expl_noise_std_hi: float = 1.0
    rew_scaling_lo: float = 1.0
    rew_scaling_hi: float = 0.1
    action_scale: float = 30

    def __post_init__(self):

        env = create_maze_env(self.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.__setattr__('state_dim', state_dim)
        self.__setattr__('action_dim', action_dim)
        # hard code goal dim to only include x, y, z
        self.__setattr__('goal_dim', 3)


class Hiro:

    def __init__(self, params: HiroConfig):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = params.env
        self.seed = params.seed
        self.state_dim = params.state_dim
        self.goal_dim = params.goal_dim
        self.act_dim = params.action_dim
        self.action_scale = params.action_scale
        self.goal_scale = get_goal_scale(params.env_name)
        self.target = get_target_position(params.env_name, device=self.device)

        self.ep_len = params.epsiod_len

        # setup seed
        self.env.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # policy parameters
        actor_lr = params.actor_lr
        critic_lr = params.critic_lr
        polyak = params.polyak
        bsize = params.bsize
        noise_std = params.policy_noise_std
        noise_clip = params.noise_clip
        discount = params.discount
        buffer_size = params.replay_buffer_size

        self.actor_lo = ActorLow(self.state_dim, self.goal_dim, self.act_dim, self.action_scale)
        self.critic_lo = CriticLow(self.state_dim, self.goal_dim, self.act_dim)
        self.replay_lo = ReplayBufferLo(self.state_dim, self.goal_dim, self.act_dim, buffer_size)
        self.agent_lo = TD3Policy(
            replay_buffer=self.replay_lo,
            actor=self.actor_lo,
            critic=self.critic_lo,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            polyak=polyak,
            bsize=bsize,
            policy_noise_std=noise_std,
            policy_noise_clip=noise_clip,
            discount=discount,
        )

        self.actor_hi = ActorHigh(self.state_dim, self.goal_dim, self.goal_scale)
        self.critic_hi = CriticHi(self.state_dim, self.goal_dim)
        self.replay_hi = ReplayBufferHi(self.state_dim, self.goal_dim, buffer_size)
        self.agent_hi = TD3Policy(
            replay_buffer=self.replay_hi,
            actor=self.actor_hi,
            critic=self.critic_hi,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            polyak=polyak,
            bsize=bsize,
            policy_noise_std=noise_std,
            policy_noise_clip=noise_clip,
            discount=discount,
        )

        # book keeping
        self.step = 0


    def correct_goal(self, init_goal: torch.Tensor, action_seq: List[torch.Tensor],
                     state_sequence: List[torch.Tensor],
                     state_end: torch.Tensor) -> [torch.Tensor, bool]:
        # state_seq (s_0, ..., s_{c-1}), act_seq (a_0, ..., a_{c-1}), state_end: s_c

        action_seq = torch.stack(action_seq, 0)
        state_sequence = torch.stack(state_sequence + [state_end], 0)

        mean = state_sequence[-1] - state_sequence[0]
        std = 0.5 * self.goal_scale
        goal_candidates = [init_goal, mean]
        goal_candidates += [torch.randn_like(mean) * std + mean for _ in range(8)]

        loss_goal = []
        for candidate in goal_candidates:
            goal_seq = (state_sequence[1:] - state_sequence[:-1])[:self.goal_dim] + candidate
            actor_state = torch.cat([state_sequence[:-1], goal_seq], dim=-1)
            hindsight_act = self.agent_lo.actor(actor_state.float())
            loss_goal.append(((action_seq - hindsight_act) ** 2).mean(-1))

        loss_goal = torch.stack(loss_goal, 0)
        index = int(torch.argmin(loss_goal))

        return goal_candidates[index]

    def evaluate(self):
        print("\n    > evaluating policies...")
        success_number = 0
        env = self.env
        target_pos = self.target.detach().cpu().numpy()
        nseed = 50
        for i in range(nseed):
            env.seed(self.seed + i)
            t = 0
            episode_len = self.ep_len
            obs, done = torch.from_numpy(env.reset()).to(self.device), False
            while not done and t < episode_len:
                goal = self.agent_hi.actor(obs)
                action = self.agent_lo.actor(obs, goal)
                obs, _, _, _ = env.step(action.detach().cpu().numpy())
                done = success_judge(obs, self.goal_dim, target_pos)
                t += 1
            if done:
                success_number += 1
            print(f"        > evaluated episodes {i} [{t} steps]")
        success_rate = success_number / nseed
        print("    > finished evaluation, success rate: {}\n".format(success_rate))
        return success_rate

    def train(self):

        max_timestep = self.params.max_timestep
        start_timestep = self.params.start_timestep
        c = self.params.hi_level_train_step
        expl_noise_lo = self.params.expl_noise_std_lo
        expl_noise_hi = self.params.expl_noise_std_hi
        bsize = self.params.bsize

        target = self.target.detach().cpu().numpy()
        env = self.env

        state = env.reset()
        goal = np.random.randn(self.goal_dim)
        act_seq, state_seq, goal_seq = [], [], []
        rew_h_accum = 0

        for t in range(self.step, max_timestep):

            if t < start_timestep:
                action = env.action_space.sample()
            else:
                action = self.agent_lo.actor(torch.from_numpy(state).float(),
                                             torch.from_numpy(goal).float())
                expl_noise = torch.randn_like(action) * expl_noise_lo
                action = torch.clamp(action + expl_noise, -self.action_scale, self.action_scale)
                action = action.detach().cpu().numpy()

            # interact with the env
            next_state, _, _, _ = env.step(action)

            rew_l = intrinsic_reward(state, goal, next_state, self.goal_dim)
            next_goal = h_function(state, goal, next_state, self.goal_dim)
            done_l = done_judge_low(goal)

            self.replay_lo.store(state, goal, action, rew_l, next_state, next_goal, done_l)

            state_seq.append(torch.from_numpy(state))
            act_seq.append(torch.from_numpy(action))
            goal_seq.append(torch.from_numpy(goal))

            rew_h = dense_reward(next_state, target, self.goal_dim)
            done_h = success_judge(next_state, target, self.goal_dim)
            rew_h_accum += rew_h

            if (t + 1) % c == 0:
                if t < start_timestep:
                    next_goal = torch.randn(self.goal_dim) * self.goal_scale
                else:
                    next_goal = self.agent_hi.actor(torch.from_numpy(state).float())
                    expl_noise = torch.randn_like(next_goal) * expl_noise_hi
                    next_goal = next_goal + expl_noise

                next_goal = torch.min(torch.max(next_goal, -self.goal_scale), self.goal_scale)
                next_goal = next_goal.detach().cpu().numpy()

                next_state_tens = torch.from_numpy(next_state)
                updated_goal = self.correct_goal(goal_seq[0], act_seq, state_seq, next_state_tens)

                self.replay_hi.store(state_seq[0], updated_goal, rew_h_accum.item(), next_state,
                                     done_h.item())

                act_seq, state_seq, goal_seq = [], [], []
                rew_h_accum = 0

            state = next_state
            goal = next_goal

            if t >= start_timestep:
                batch = self.replay_lo.sample_batch(bsize, return_tensor=True, device=self.device)
                self.agent_lo.step_update()









