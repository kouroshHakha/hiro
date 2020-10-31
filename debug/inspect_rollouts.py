import torch
import argparse
import numpy as np
from sklearn.decomposition.pca import PCA
from sklearn.manifold.t_sne import TSNE
if torch.cuda.is_available():
    from tsnecuda import TSNE

import time

import matplotlib.pyplot as plt

from debug.cpdb import register_pdb
register_pdb()

def _parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    return parser.parse_args()


def plot_actions(actions):
    rollouts = []
    ep_ids = []
    for ep_idx, ep in enumerate(actions):
        ep_actions = np.stack(ep, axis=0)
        if ep_actions.ndim == 3:
            ep_actions = ep_actions.squeeze(-2)
        rollouts.append(ep_actions)
        ep_ids += [ep_idx] * len(ep_actions)

    ep_ids = np.array(ep_ids)
    rollouts = np.concatenate(rollouts, axis=0)
    # rollouts_2d = PCA(n_components=2).fit_transform(rollouts)
    print(f'fiting tsne on {len(rollouts)} points ...')
    s = time.time()
    rollouts_2d = TSNE(n_components=2).fit_transform(rollouts)
    print(f'finished in {time.time() - s}')

    img = plt.scatter(rollouts_2d[:, 0], rollouts_2d[:, 1], c=ep_ids, cmap='viridis', alpha=0.5)
    plt.colorbar(img)

    plt.savefig('action.png', dpi=300)

    breakpoint()


def plot_states(states):
    n = 100
    m = 10
    neps = len(states)

    nindices = np.arange(0, neps, neps / n).astype(int)
    sel_states = [[states[i][j] for j in np.arange(0, len(states[i]), len(states[i]) / m).astype(int)] for i in nindices]
    sel_states = np.array(sel_states)

    colors = plt.cm.viridis(np.linspace(0.1,0.9,len(sel_states)))
    for ep, color in zip(sel_states, colors):
        plt.plot(ep[:, 0], ep[:, 1], color=color)
    plt.show()
    breakpoint()




if __name__ == '__main__':
    _args = _parse_arg()

    ckpt = torch.load(_args.checkpoint, map_location='cpu')
    rollouts = ckpt['rollouts']

    # plot_actions(rollouts['actions'])
    plot_states(rollouts['states'])
