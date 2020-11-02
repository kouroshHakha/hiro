"""
Unittest for TD3 implementation
"""

import gym
from gym import wrappers
import time
import datetime
import dataclasses
import torch
import numpy as np
from pathlib import Path

from logger import VectorLogger
from replay import ReplayTD3, BatchT
from network import ActorTD3, CriticTD3
from policy import TD3Policy

from debug.cpdb import register_pdb
register_pdb()


class ReplayBuffer(ReplayTD3):

    def __init__(self, obs_dim: int, act_dim: int, size: int = 200000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray,
              done: float):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size: int = 32, return_tensor: bool = False,
                     device: torch.device = torch.device('cpu')) -> BatchT:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=self.obs_buf[idxs],
            next_state=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs]
        )

        if return_tensor:
            return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}
        else:
            return batch


@dataclasses.dataclass
class TestConfig:
    env_name: str = 'InvertedPendulum-v2'
    seed: int = 0
    gamma: float = 0.99
    epochs: int = 100
    polyak: float = 0.995
    steps_per_epoch: int = 4000
    replay_size: int = int(1e6)
    pi_lr: float= 1e-3
    q_lr: float = 1e-3
    batch_size: int = 100
    start_steps: int = 10000
    update_after: int = 1000
    update_every: int = 50
    act_noise: float = 0.1
    target_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    num_test_episodes: int = 10
    max_ep_len: int = 1000
    save_freq: int = 1
    save_video_every: int = 1

class TestTD3:

    def __init__(self, config: TestConfig = None):

        if config is None:
            config = TestConfig()

        self.params = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make(self.params.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high

        # setup seed
        self.seed = self.params.seed
        self._seed()

        self.actor = ActorTD3(self.obs_dim, self.act_dim, self.act_limit)
        self.critic = CriticTD3(self.obs_dim, self.act_dim)
        self.replay = ReplayBuffer(self.obs_dim, self.act_dim, self.params.replay_size)
        self.policy = TD3Policy(
            replay_buffer=self.replay,
            actor=self.actor,
            critic=self.critic,
            actor_lr=self.params.pi_lr,
            critic_lr=self.params.q_lr,
            polyak=self.params.polyak,
            bsize=self.params.batch_size,
            policy_noise_std=self.params.target_noise,
            policy_noise_clip=self.params.noise_clip,
            discount=self.params.gamma,
            device=self.device,
        )

        self.path = Path(f'/tmp/experiments/unittest_hiro/{self.params.env_name}/s{self.seed}')
        self.logger = VectorLogger(output_dir=str(self.path))

        # Set up model saving
        self.logger.setup_pytorch_saver(self.actor)


    def get_action(self, o, noise_scale):
        a = self.actor(torch.as_tensor(o, dtype=torch.float32).to(self.device))
        a = a.detach().cpu().numpy()
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def run_agent(self, nepisodes, render: bool = False):
        env = self.env
        if render:
            stamp = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            env = wrappers.Monitor(env, str(self.path / 'videos' / stamp))

        for j in range(nepisodes):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not(d or (ep_len == self.params.max_ep_len)):
                if render:
                    env.render()
                    time.sleep(1e-3)
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


    def test(self):

        logger = self.logger
        total_steps = self.params.steps_per_epoch * self.params.epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for t in range(total_steps):

            if t > self.params.start_steps:
                a = self.get_action(o, self.params.act_noise)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1


            d = False if ep_len == self.params.max_ep_len else d

            self.replay.store(o, a, r, o2, d)

            o = o2

            # End of trajectory handling
            if d or (ep_len == self.params.max_ep_len):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            if t >= self.params.update_after and t % self.params.update_every == 0:
                for j in range(self.params.update_every):
                    batch = self.replay.sample_batch(self.params.batch_size, return_tensor=True,
                                                     device=self.device)
                    info = self.policy.step_update(batch, j % self.params.policy_delay == 0)
                    logger.store(Q1Vals=info['q1_val'], Q2Vals=info['q2_val'], LossQ=info['q_loss'])
                    if 'pi_loss' in info:
                        logger.store(LossPi=info['pi_loss'])



            if (t + 1) % self.params.steps_per_epoch == 0:

                epoch = (t+1) // self.params.steps_per_epoch

                # Save model
                if (epoch % self.params.save_freq == 0) or (epoch == self.params.epochs):
                    logger.save_state({'env': self.env}, None)
                # Test the performance of the deterministic version of the agent.
                render = epoch % self.params.save_video_every == 0
                nepisodes = self.params.num_test_episodes
                if render:
                    nepisodes = min(10, nepisodes)
                self.run_agent(nepisodes, render=render)

                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()

    def _seed(self):
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='InvertedPendulum-v2')
    args = parser.parse_args()

    unit = TestTD3(TestConfig(env_name=args.env))
    unit.test()