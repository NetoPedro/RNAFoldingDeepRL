import policy as policy_module
import env_rna
import torch
import numpy as np

env = env_rna.EnvRNA()
def train():
    # TODO Reformulate as ASC3 algorithm
    policy = policy_module.Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    running_reward = 0
    MAX_ITER = 300
    N = 10
    alpha = .95

    for i_episode in range(1):
        seq = generate_random_sequence(N)
        state = env.reset(seq)
        best_reward = 0

        for t in range(MAX_ITER):
            rewards = []
            action = 0 # select_action(policy,state)
            state, reward, done, _ = env.step(action,N)
            rewards.append(reward)
            reward = max(reward,best_reward)
            if done:
                break

        running_reward = running_reward * alpha + best_reward * (1-alpha)

        # TODO finish episode MC or TD to update the policy. Alternative?
        # finish_episode(optimizer, rewards)



def generate_random_sequence(N):
    sequence = ""
    for i in range (N):
        epsilon = np.random.uniform(0,1)
        if epsilon <= 0.25:
            sequence += "A"
        elif epsilon <= 0.5:
            sequence += "U"
        elif epsilon <= 0.75:
            sequence+= "G"
        else:
            sequence+= "C"

    return sequence

