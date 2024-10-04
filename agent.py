import torch
import numpy as np


class ActorCritic(torch.nn.Module):
    def __init__(self, state_size, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()
        self.gamma = gamma

        self.conv1 = torch.nn.Conv2d(
            state_size[0], 32, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.conv_shape = self._calculate_conv_shape(state_size)

        self.gru = torch.nn.GRUCell(self.conv_shape, 256)
        self.pi = torch.nn.Linear(256, n_actions)
        self.v = torch.nn.Linear(256, 1)

    def _calculate_conv_shape(self, state_size):
        o = self.conv1(torch.zeros(1, *state_size))
        o = self.conv2(o)
        o = self.conv3(o)
        o = self.conv4(o)
        return int(np.prod(o.size()))

    def forward(self, x, hidden_state):
        x = torch.nn.functional.elu(self.conv1(x))
        x = torch.nn.functional.elu(self.conv2(x))
        x = torch.nn.functional.elu(self.conv3(x))
        x = torch.nn.functional.elu(self.conv4(x))
        x = x.view(x.size(0), -1)

        hidden_state = self.gru(x, (hidden_state))

        pi = self.pi(hidden_state)
        v = self.v(hidden_state)

        probs = torch.nn.functional.softmax(pi, dim=-1)
        dist = torch.distributions.Categorical(probs)  # discrete action space
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], log_prob, v, hidden_state


if __name__ == "__main__":
    state_size = (4, 42, 42)
    n_actions = 6
    model = ActorCritic(state_size, n_actions)
    x = torch.randn(1, *state_size)
    hidden_state = torch.zeros(1, 256)
    action, log_prob, v, hidden_state = model(x, hidden_state)
    print(action, log_prob, v, hidden_state)
