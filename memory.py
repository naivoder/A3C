class Memory:
    def __init__(self):
        self.rewards = []
        self.values = []
        self.log_probs = []

    def remember(self, reward, value, log_prob):
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear(self):
        self.rewards = []
        self.values = []
        self.log_probs = []

    def sample(self):
        return self.rewards, self.values, self.log_probs
