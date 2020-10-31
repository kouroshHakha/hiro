from typing import List, Tuple, Dict, Any, Optional


import torch
import numpy as np
import dataclasses
import wandb
from pathlib import Path

from policy import TD3Policy
from network import ActorLow, CriticLow, ActorHigh, CriticHi
from replay import ReplayBufferHi, ReplayBufferLo
from environments.create_maze_env import create_maze_env
from util import (
    get_goal_scale, get_target_position, success_judge, h_function, intrinsic_reward,
    dense_reward, done_judge_low
)


@dataclasses.dataclass
class HiroConfig:
    env_name: str = 'AntPush'
    seed: int = 0
    max_timestep: int = 10e6
    start_timestep: int = 100e3
    episode_len: int = 1000
    batch_size: int = 32
    c: int = 10
    policy_freq: int = 1
    discount: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    polyak: float = 0.995
    rew_scaling_lo: float = 1.0
    rew_scaling_hi: float = 0.1
    policy_noise_std: float = 1.0
    policy_noise_clip: float = 1.0
    expl_noise_std_lo: float = 1.0
    expl_noise_std_hi: float = 1.0
    action_scale: float = 30
    replay_buffer_size: int = 200000

    # logging params
    save_video: bool = True
    video_interval: int = 50000
    checkpoint_interval: int = 10000
    evaluation_interval: int = 50000
    log_interval: int = 100
    checkpoint: str = ''
    prefix: str = ''
    _state_dict: Dict = dataclasses.field(default_factory=dict, repr=False, compare=False)


    def __post_init__(self):
        for field in dataclasses.fields(self):
            if field.name != '_state_dict':
                value = getattr(self, field.name)
                if not isinstance(value, field.type):
                    object.__setattr__(self, field.name, field.type(value))
                self._state_dict[field.name] = getattr(self, field.name)

        self._set_env()

    def state_dict(self) -> Dict[str, Any]:
        return self._state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for k, v in state_dict.items():
            object.__setattr__(self, k, v)
        self._set_env()

    def _set_env(self):
        env = create_maze_env(self.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        object.__setattr__(self, 'env', env)
        object.__setattr__(self, 'state_dim', state_dim)
        object.__setattr__(self, 'action_dim', action_dim)
        # hard code goal dim to only include x, y, z
        object.__setattr__(self, 'goal_dim', 3)


class Hiro:

    def __init__(self, params: HiroConfig):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        if params.checkpoint:
            self.params = self.load_checkpoint(params.checkpoint, get_params_only=True)

        wandb.config.update(self.params.state_dict())

        self.env = self.params.env
        self.seed = self.params.seed
        self.state_dim = self.params.state_dim
        self.goal_dim = self.params.goal_dim
        self.act_dim = self.params.action_dim
        self.action_scale = self.params.action_scale
        self.goal_scale = get_goal_scale(self.params.env_name, device=self.device)
        self.target = get_target_position(self.params.env_name, device=self.device)
        self.rew_scaling_lo = self.params.rew_scaling_lo
        self.rew_scaling_hi = self.params.rew_scaling_hi

        self.ep_len = self.params.episode_len

        # setup seed
        self._seed()

        # policy parameters
        actor_lr = self.params.actor_lr
        critic_lr = self.params.critic_lr
        polyak = self.params.polyak
        bsize = self.params.batch_size
        noise_std = self.params.policy_noise_std
        noise_clip = self.params.policy_noise_clip
        discount = self.params.discount
        buffer_size = self.params.replay_buffer_size

        self.actor_lo: ActorLow = ActorLow(self.state_dim, self.goal_dim, self.act_dim,
                                           self.action_scale)
        self.critic_lo: CriticLow = CriticLow(self.state_dim, self.goal_dim, self.act_dim)
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
            device=self.device,
        )

        self.actor_hi: ActorHigh = ActorHigh(self.state_dim, self.goal_dim, self.goal_scale)
        self.critic_hi: CriticHi = CriticHi(self.state_dim, self.goal_dim)
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
            device=self.device
        )

        # book keeping
        self.step = 0
        self.ep_idx = 0
        self.rollouts = dict(states=[], actions=[], goals=[], rewards=[])

        if params.checkpoint:
            self.load_checkpoint(params.checkpoint)


    def correct_goal(self, init_goal: torch.Tensor, action_seq: List[torch.Tensor],
                     state_seq: List[torch.Tensor],
                     state_end: torch.Tensor) -> [torch.Tensor, bool]:
        # state_seq (s_0, ..., s_{c-1}), act_seq (a_0, ..., a_{c-1}), state_end: s_c
        action_seq = torch.stack(action_seq, 0)
        state_seq = torch.stack(state_seq + [state_end], 0)

        mean = (state_seq[-1] - state_seq[0])[:self.goal_dim]
        std = 0.5 * self.goal_scale
        goal_candidates = [init_goal]
        # TODO: diff (forgot to clamp the new candidates)
        for _ in range(8):
            candidate = torch.randn_like(mean) * std + mean
            candidate = torch.min(torch.max(candidate, -self.goal_scale), self.goal_scale)
            goal_candidates.append(candidate)
        goal_candidates += [mean]

        loss_goal = []
        # TODO: Why not just use mean and init_goal, Does this randomization actually play a significant role?
        for idx, candidate in enumerate(goal_candidates):
            goal_seq = (state_seq[0] - state_seq[:-1])[:, :self.goal_dim] + candidate
            actor_state = torch.cat([state_seq[:-1], goal_seq], dim=-1)
            hindsight_act = self.agent_lo.actor(actor_state)
            loss_goal.append(((action_seq - hindsight_act) ** 2).mean())

        loss_goal = torch.stack(loss_goal, 0)
        index = int(torch.argmin(loss_goal))
        updated = (index != 0)

        return goal_candidates[index], updated

    def evaluate(self):
        print("\n    > evaluating policies...")
        success_number = 0
        env = self.env
        nseed = 50
        for i in range(nseed):
            env.seed(self.seed + i)
            done, ep_reward = self._run_episode(render=False)
            if done:
                success_number += 1
            print(f"        > evaluated episodes {i} ep_reward = {ep_reward:.3f}")
        success_rate = success_number / nseed
        print(f"    > finished evaluation, success rate: {success_rate:.2f}\n")
        return success_rate

    def log_video_hrl(self):
        print('\n    > Collecting current trajectory...')
        _, ep_reward, frame_buffer = self._run_episode(render=True)
        print(f'    > Finished collection, saved video. Episode reward: {float(ep_reward):.3f}\n')
        frame_buffer = np.array(frame_buffer).transpose([0, 3, 1, 2])

        wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})

    def _run_episode(self, render=False):
        frame_buffer= []
        target = self.target.detach().cpu().numpy()
        t, ep_reward = 0, 0
        state, done = self.env.reset(), False

        # extra variables for proper execution (initial value not important)
        goal = None

        while not done and t < self.ep_len:
            if render:
                frame_buffer.append(self.env.render(mode='rgb_array'))
            state_tens = torch.from_numpy(state).float().to(self.device)
            # TODO: run episodes exactly like we collect experience
            if (t + 1) % self.params.c == 0:
                goal = self.agent_hi.actor(state_tens).squeeze(0)

            action = self.agent_lo.actor(torch.cat([state_tens, goal], dim=-1))
            next_state, _, _, _ = self.env.step(action.detach().cpu().numpy())
            reward = dense_reward(next_state, target, self.goal_dim)
            done = success_judge(next_state, target, self.goal_dim)
            next_state_tens = torch.from_numpy(next_state).float().to(self.device)

            t += 1
            ep_reward += reward
            goal = goal + (state_tens - next_state_tens)[:self.goal_dim]
            state = next_state

        if render:
            return done, ep_reward, frame_buffer

        return done, ep_reward


    def train(self):

        max_timestep = self.params.max_timestep
        start_timestep = self.params.start_timestep
        c = self.params.c
        expl_noise_lo = self.params.expl_noise_std_lo
        expl_noise_hi = self.params.expl_noise_std_hi
        bsize = self.params.batch_size
        policy_freq = self.params.policy_freq

        target = self.target.detach().cpu().numpy()
        env = self.env

        # logging and book keeping
        ep_idx, ep_t_idx = self.ep_idx, 0
        ep_reward_l, ep_reward_h = 0, 0
        ep_rews, ep_states, ep_actions, ep_goals = [], [], [], []
        checkpoint_event = False

        state, goal = self._reset(t=-1)

        act_seq, state_seq, goal_seq = [], [], []
        rew_h_accum = 0.0
        agent_lo_iter, agent_hi_iter = 0, 0

        for t in range(self.step, max_timestep):
            if t < start_timestep:
                action = env.action_space.sample()
            else:
                lo_state = torch.cat([torch.from_numpy(state).float(),
                                      torch.from_numpy(goal).float()], dim=-1).to(self.device)
                action = self.agent_lo.actor(lo_state).squeeze(0)
                expl_noise = torch.randn_like(action) * expl_noise_lo
                action = torch.clamp(action + expl_noise, -self.action_scale, self.action_scale)
                action = action.detach().cpu().numpy()

            # interact with the env
            next_state, _, _, _ = env.step(action)

            rew_l = self.rew_scaling_lo * intrinsic_reward(state, goal, next_state,
                                                           self.goal_dim).item()
            next_goal = h_function(state, goal, next_state, self.goal_dim)
            # on low level we are done when next_goal is small
            done_l = float(done_judge_low(next_goal).item())

            self.replay_lo.store(state, goal, action, rew_l, next_state, next_goal, done_l)

            state_seq.append(torch.from_numpy(state).float().to(self.device))
            act_seq.append(torch.from_numpy(action).float().to(self.device))
            goal_seq.append(torch.from_numpy(goal).float().to(self.device))

            rew_h = dense_reward(next_state, target, self.goal_dim)
            done_h = float(success_judge(next_state, target, self.goal_dim).item())
            rew_h_accum += self.rew_scaling_hi * rew_h.item()

            # book keeping
            ep_states.append(state)
            ep_actions.append(action)
            ep_goals.append(goal)
            ep_rews.append(rew_h)

            # # TODO: I wonder if we add done_h to the high level condition it makes any difference?
            if (t + 1) % c == 0:
                next_state_tens = torch.from_numpy(next_state).float().to(self.device)
                if t < start_timestep:
                    next_goal = torch.randn_like(self.goal_scale) * self.goal_scale
                else:
                    # TODO: diff (big bug: was using state instead of next_state)
                    next_goal = self.agent_hi.actor(next_state_tens)
                    expl_noise = torch.randn_like(next_goal) * expl_noise_hi
                    next_goal = next_goal + expl_noise

                next_goal = torch.min(torch.max(next_goal, -self.goal_scale), self.goal_scale)

                # relabel and store experience in the buffer
                updated_goal, updated = self.correct_goal(goal_seq[0], act_seq, state_seq,
                                                          next_state_tens)
                updated_goal = updated_goal.detach().cpu().numpy()
                state_start = state_seq[0].detach().cpu().numpy()

                wandb.log(dict(rew_h_accum=rew_h_accum), step=t)
                self.replay_hi.store(state_start, updated_goal, rew_h_accum, next_state, done_h)

                # reset values
                act_seq, state_seq, goal_seq = [], [], []
                rew_h_accum = 0.0

                # shape is (1 x D)
                next_goal = next_goal.squeeze(0).detach().cpu().numpy()

            state = next_state
            goal = next_goal

            if t >= start_timestep:
                batch = self.replay_lo.sample_batch(bsize, return_tensor=True, device=self.device)
                agent_lo_iter += 1
                low_loss_info = self.agent_lo.step_update(batch, agent_lo_iter % policy_freq == 0)
                wandb.log({'low': low_loss_info}, step=t)

            if t >= start_timestep and (t + 1) % c == 0:
                batch = self.replay_hi.sample_batch(bsize, return_tensor=True, device=self.device)
                agent_hi_iter += 1
                hi_loss_info = self.agent_hi.step_update(batch, agent_hi_iter % policy_freq == 0)
                wandb.log({'high': hi_loss_info}, step=t)

            ep_reward_l += rew_l
            ep_reward_h += rew_h
            ep_t_idx += 1

            # book keeping and evaluation
            if (t + 1) % self.params.log_interval == 0:
                avg_rew_per_step = np.mean(ep_rews)
                print(f'[step {t + 1}] ep_id = {ep_idx}, ep_t_idx = {ep_t_idx}, '
                      f'avg_rew_per_step = {avg_rew_per_step:.3f}')

            # at the end of an episode reset env and store episode reward
            episode_end = ep_t_idx >= self.ep_len or done_h
            if episode_end:
                wandb.log(dict(ep_reward_l=ep_reward_l, ep_reward_h=ep_reward_h), step=t)

                self.rollouts['states'].append(ep_states)
                self.rollouts['actions'].append(ep_actions)
                self.rollouts['goals'].append(ep_goals)
                self.rollouts['rewards'].append(ep_rews)

                ep_rews, ep_states, ep_actions, ep_goals = [], [], [], []
                ep_t_idx, ep_reward_l, ep_reward_h = 0, 0, 0
                ep_idx += 1
                state, goal = self._reset(t)

            if (t + 1) % self.params.evaluation_interval == 0:
                success_rate = self.evaluate()
                wandb.log({'success_rate': success_rate}, step=t)

            if t == 0 or (t + 1) % self.params.video_interval == 0 and self.params.save_video:
                self.log_video_hrl()

            # should save the checkpoint exactly at the end of previous episode
            if not checkpoint_event:
                checkpoint_event = (t + 1) % self.params.checkpoint_interval == 0
            if checkpoint_event and episode_end:
                # should start from the next step when we come in to the loop after loading
                # ep_idx is already incremented
                self.save_checkpoint(t + 1, ep_idx)
                checkpoint_event = False


    def save_checkpoint(self, step, ep_idx):
        env_name = self.params.env_name.lower()
        prefix = self.params.prefix

        file_name = f'checkpoint-hiro-{env_name}'
        if prefix:
            file_name = f'{file_name}-{prefix}'
        file_name = f'{file_name}.tar'

        file_path = Path('.') / 'save' / 'model' / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = dict(
            step=step,
            ep_idx=ep_idx,
            params=self.params.state_dict(),
            rollouts=self.rollouts,
            actor_lo=self.actor_lo.state_dict(),
            critic_lo=self.critic_lo.state_dict(),
            replay_lo=self.replay_lo,
            actor_lo_optimizer=self.agent_lo.actor_optim.state_dict(),
            critic_lo_optimizer=self.agent_lo.critic_optim.state_dict(),
            actor_hi=self.actor_hi.state_dict(),
            critic_hi=self.critic_hi.state_dict(),
            replay_hi=self.replay_hi,
            actor_hi_optimizer=self.agent_hi.actor_optim.state_dict(),
            critic_hi_optimizer=self.agent_hi.critic_optim.state_dict(),
        )

        torch.save(save_dict, file_path)
        print(f"    > saved checkpoint to: {str(file_path)}\n")

    def load_checkpoint(self, checkpoint_path: str,
                        get_params_only: bool = False) -> Optional[HiroConfig]:
        if not get_params_only:
            print("\n    > loading training checkpoint...")
        load_dict = torch.load(checkpoint_path, map_location=self.device)
        if not get_params_only:
            print("\n    > checkpoint file loaded! parsing data...")
        if get_params_only:
            params = HiroConfig()
            params.load_state_dict(load_dict['params'])
            return params

        self.step = load_dict['step']
        self.ep_idx = load_dict['ep_idx']
        self.rollouts = load_dict['rollouts']

        self.actor_lo.load_state_dict(load_dict['actor_lo'])
        self.critic_lo.load_state_dict(load_dict['critic_lo'])
        self.replay_lo = load_dict['replay_lo']
        self.agent_lo.actor_optim.load_state_dict(load_dict['actor_lo_optimizer'])
        self.agent_lo.critic_optim.load_state_dict(load_dict['critic_lo_optimizer'])

        self.actor_hi.load_state_dict(load_dict['actor_hi'])
        self.critic_hi.load_state_dict(load_dict['critic_hi'])
        self.replay_hi = load_dict['replay_hi']
        self.agent_hi.actor_optim.load_state_dict(load_dict['actor_hi_optimizer'])
        self.agent_hi.critic_optim.load_state_dict(load_dict['critic_hi_optimizer'])

        print("    > checkpoint resume success!")

    def _compute_goal_seq(self, init_goal: torch.Tensor,
                          state_sequence: torch.Tensor) -> torch.Tensor:
        # state ~ (C x sdim), goal_seq ~ (C x goal_dim)
        goal_seq: List[torch.Tensor] = [init_goal]
        for next_state, prev_state in zip(state_sequence[1:], state_sequence[:-1]):
            cur_goal = (prev_state - next_state)[:self.goal_dim] + goal_seq[-1]
            goal_seq.append(cur_goal)

        # sanity check
        if len(goal_seq) != len(state_sequence):
            raise ValueError(f'goal sequence length mismatch with state sequence, '
                             f'{len(goal_seq)} != {len(state_sequence)}')

        return torch.stack(goal_seq, dim=0)


    def _seed(self):
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


    def _reset(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        goal_scale = self.goal_scale.detach().cpu().numpy()
        state = self.env.reset()

        if t < self.params.start_timestep:
            # TODO: diff
            # goal = np.random.randn(self.goal_dim) * goal_scale
            goal = np.random.randn(self.goal_dim)
            # clip goal to the scale
            goal = np.clip(goal, -goal_scale, goal_scale)
        else:
            goal = self.agent_hi.actor(torch.from_numpy(state).float().to(self.device))
            goal = goal.squeeze(0).detach().cpu().numpy()

        return state, goal
