import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv
from ale_py import ALEInterface, LoggerMode
from config import environments
import warnings
import gymnasium as gym

warnings.simplefilter("ignore")
ALEInterface.setLoggerMode(LoggerMode.Error)

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "--n_threads",
        default=4,
        type=int,
        help="Number of parallel environments during training",
    )
    parser.add_argument(
        "--n_games",
        default=1000,
        type=int,
        help="Total number of episodes (games) to play during training",
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights", "csv"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    input_shape = (4, 42, 42)

    if args.env:
        config_env = gym.make(args.env)
        n_actions = config_env.action_space.n

        print("Environment:", args.env)
        print("Observation space:", config_env.observation_space)
        print("Action space:", config_env.action_space)

        mp.set_start_method("forkserver")
        env = ParallelEnv(args.env, args.n_threads, input_shape, n_actions)
    else:
        for env_name in environments:
            args.env = env_name
            config_env = gym.make(args.env)
            n_actions = config_env.action_space.n

            print("Environment:", args.env)
            print("Observation space:", config_env.observation_space)
            print("Action space:", config_env.action_space)

            mp.set_start_method("forkserver")
            env = ParallelEnv(args.env, args.n_threads, input_shape, n_actions)