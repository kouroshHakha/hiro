"""
CUDA_VISIBLE_DEVICES=1,2 taskset -c 38-48 python run.py --env_name AntPush --max_timestep 1000000 --start_timestep 50000 --checkpoint_interval 50000 --save_video
"""
import argparse
import csv
import wandb
import os

from debug.cpdb import register_pdb

from hiro import Hiro, HiroConfig

register_pdb()

def _parse_args():
    # =============================
    # Command-Line Argument Binding
    # =============================
    parser = argparse.ArgumentParser(description="Specific Hyper-Parameters for HIRO training. ")
    # >> experiment parameters
    parser.add_argument('--param_id', default=1, type=int, help='index of parameter that will be loaded from local csv file for this run')
    parser.add_argument('--env_name', help='environment name for this run, choose from AntPush, AntFall, AntMaze')
    # >> HIRO alg parameters
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--max_timestep', type=int, help='max training time step')
    parser.add_argument('--start_timestep', type=int, help='amount of random filling experience')
    parser.add_argument('--episode_len', type=int, help='max number of time steps of an episode')
    parser.add_argument('--batch_size', type=int, help='batch sample size')
    parser.add_argument('--c', type=int, help='high-level policy update interval')
    parser.add_argument('--policy_freq', type=int, help='delayed policy update interval')
    parser.add_argument('--discount', type=float, help='long-horizon reward discount')
    parser.add_argument('--actor_lr', type=float, help='actor policy learning rate')
    parser.add_argument('--critic_lr', type=float, help='critic policy learning rate')
    parser.add_argument('--polyak', type=float, help='soft update parameter')
    parser.add_argument('--rew_scaling_lo', type=float, help='low-level reward rescale parameter')
    parser.add_argument('--rew_scaling_hi', type=float, help='high-level reward rescale parameter')
    parser.add_argument('--policy_noise_std', type=float, help='target policy smoothing regularization noise standard deviation')
    parser.add_argument('--policy_noise_clip', type=float, help='exploration noise boundary')
    parser.add_argument('--expl_noise_std_lo', type=float, help='low-level policy exploration noise standard deviation')
    parser.add_argument('--expl_noise_std_hi', type=float, help='low-level policy exploration noise standard deviation')
    parser.add_argument('--action_scale', type=int, help='action boundary')
    parser.add_argument('--replay_buffer_size', type=int, help='The size of the replay buffer for both controllers')
    # >> logging params
    parser.add_argument('--save_video', action='store_true', help='whether sample and log episode video intermittently during training')
    parser.add_argument('--video_interval', type=int, help='the interval of logging video')
    parser.add_argument('--checkpoint_interval', type=int, help='the interval of logging checkpoint')
    parser.add_argument('--evaluation_interval', type=int, help='the interval of logging evaluation utils')
    parser.add_argument('--log_interval', type=int, help='the interval of print training state to interval')
    parser.add_argument('--checkpoint', help='the file name of checkpoint to be load, set to None if do not load data from local checkpoint')
    parser.add_argument('--prefix', help='prefix of checkpoint files, used to distinguish different runs')
    # >> parse arguments
    args = parser.parse_args()
    return args

def load_params(cmd_args: argparse.Namespace) -> HiroConfig:
    fpath = './hiro.csv'
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            rows = list(csv.DictReader(f))
            file_param = rows[cmd_args.param_id - 1]
    else:
        file_param = {}

    override_dict = {k: v for k, v in vars(cmd_args).items() if v is not None}
    file_param.update(override_dict)

    if 'param_id' in file_param:
        del file_param['param_id']

    return HiroConfig(**file_param)

if __name__ == '__main__':
    _args = _parse_args()
    params = load_params(_args)
    wandb.init(project="hiro_ours")
    hiro = Hiro(params)
    hiro.train()