import torch.multiprocessing as mp
from agent import ActorCritic
from shared_adam import SharedAdam

from worker import worker


class ParallelEnv:
    def __init__(self, env_id, n_threads, input_shape, n_actions, global_ep):
        names = [str(i) for i in range(n_threads)]

        global_agent = ActorCritic(input_shape, n_actions)
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
                    global_ep,
                ),
            )
            for name in names
        ]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]
