import policy as policy_module
import env_rna
import torch
import numpy as np
from torch.distributions import Categorical

env = env_rna.EnvRNA()
def train():
    # TODO Reformulate as A3C algorithm

    running_reward = 0
    MAX_ITER = 300
    N = 10
    alpha = .95

    policy = policy_module.Policy(N)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)


    for i_episode in range(1):
        seq = generate_random_sequence(N)
        env.reset(seq)
        state = env.rna.structure_representation
        best_reward = 0

        for t in range(MAX_ITER):
            rewards = []
            action =  select_action(state,policy)
            state, reward, done, _ = env.step(action,N)
            rewards.append(reward)
            reward = max(reward,best_reward)
            if done:
                break

        running_reward = running_reward * alpha + best_reward * (1-alpha)

        # TODO finish episode MC or TD to update the policy. Alternative?
        # finish_episode(optimizer, rewards)



def select_action(state,policy):
    probs = policy.forward(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

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

