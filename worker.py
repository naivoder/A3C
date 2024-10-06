import numpy as np
import torch
from memory import Memory
from agent import ActorCritic
from utils import plot_learning_curve
from wrappers import make_env


def worker(name, input_shape, n_actions, global_agent, optimizer, env_id, global_idx):
    T_MAX = 20
    local_agent = ActorCritic(input_shape, n_actions)
    memory = Memory()
    frame_buffer = [input_shape[1], input_shape[2], 1]
    env = make_env(env_id, frame_buffer)

    episode, max_eps, t_steps, scores = 0, 1000, 0, []

    while episode < max_eps:
        state, _ = env.reset()
        score, ep_steps = 0, 0
        term = trunc = False
        hx = torch.zeros(1, 256)

        while not term or trunc:
            state = torch.tensor(np.array(state), dtype=torch.float)
            action, value, log_prob, hx = local_agent(state, hx)
            state_, reward, term, trunc, _ = env.step(action)
            memory.remember(reward, value, log_prob)
            score += reward
            state = state_
            ep_steps += 1
            t_steps += 1

            if ep_steps % T_MAX == 0 or term or trunc:
                rewards, values, log_probs = memory.sample()
                loss = local_agent.calculate_cost(
                    state_, hx, rewards, values, log_probs, term or trunc
                )
                optimizer.zero_grad()
                hx = hx.detach()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                for local_param, global_param in zip(
                    local_agent.parameters(), global_agent.parameters()
                ):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())
                memory.clear()

        episode += 1
        with global_idx.get_lock():
            global_idx.value += 1

        if name == "1":
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print(f"Episode: {episode}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

    if name == "1":
        x = [i for i in range(episode)]
        plot_learning_curve(x, scores, "a3c.png")
