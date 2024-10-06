import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    mp.set_start_method("forkserver")
    gloabl_ep = mp.Value("i", 0)

    env_id = "PongNoFrameskip-v4"
    n_threads = 8
    n_actions = 6
    input_shape = (4, 84, 84)

    env = ParallelEnv(env_id, n_threads, input_shape, n_actions, gloabl_ep)
