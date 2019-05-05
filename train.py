import policy as policy_module
import env_rna
import torch
import numpy as np
from torch.distributions import Categorical

env = env_rna.EnvRNA()

class MonteCarloReinforceTrainer:

    def train(self):
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
            ep_reward = 0

            for t in range(MAX_ITER):
                rewards = []
                action =  select_action_reinforce(state,policy)
                state, reward, done, _ = env.step(action,N)
                rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            running_reward = running_reward * alpha + ep_reward * (1-alpha)

            self.finish_episode(optimizer, rewards, policy)



    def finish_episode(self, optimizer, rewards, policy):
        R = 0
        DISCOUNT_FACTOR = 0.99
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r+DISCOUNT_FACTOR*R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std())
        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.rewards[:]
        del policy.saved_log_probs[:]




def select_action_reinforce(state,policy):
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




