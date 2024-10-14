import torch.multiprocessing as mp
from agent import AC3
from shared_adam import SharedAdam

from worker import worker


class ParallelEnv:
    def __init__(self, env_id, n_threads, input_shape, n_actions, n_games):
        names = [str(i) for i in range(n_threads)]

        global_agent = AC3(input_shape, n_actions)
        global_agent.share_memory()  # ???

        optimizer = SharedAdam(global_agent.parameters(), lr=1e-4)
        self.ps = [
            mp.Process(
                target=worker,
                args=(
                    name,
                    input_shape,
                    n_actions,
                    global_agent,
                    optimizer,
                    env_id,
                    n_games
                ),
            )
            for name in names
        ]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]
