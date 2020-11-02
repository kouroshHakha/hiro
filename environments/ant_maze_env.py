from environments.maze_env import MazeEnv
from environments.ant import AntEnv


class AntMazeEnv(MazeEnv):
    metadata = {'render.modes': ['rgb_array']}
    MODEL_CLASS = AntEnv
