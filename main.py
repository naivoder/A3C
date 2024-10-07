import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    mp.set_start_method("forkserver")
    gloabl_ep = mp.Value("i", 0)

    env_id = "MsPacmanNoFrameskip-v4"
    n_threads = 4
    n_actions = 9  # need to change this to be dynamic
    input_shape = (4, 42, 42)

    env = ParallelEnv(env_id, n_threads, input_shape, n_actions, gloabl_ep)
