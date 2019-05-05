import policy as policy_module
import env_rna
import torch
import numpy as np
from torch.distributions import Categorical

env = env_rna.EnvRNA()

class Reinforce:


    running_reward = 0
    MAX_ITER = 300
    N = 10
    alpha = .95

    policy = policy_module.Policy(N)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    def select_action(self,state, policy):
        probs = policy.forward(state)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

class MonteCarloReinforceTrainer(Reinforce):

    def train(self):




        for i_episode in range(1):
            seq = generate_random_sequence(self.N)
            env.reset(seq)
            state = env.rna.structure_representation
            ep_reward = 0

            for t in range(self.MAX_ITER):
                rewards = []
                action = self.select_action(convert_to_tensor(state, seq), self.policy)
                state, reward, done, _ = env.step(action,self.N)
                rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            running_reward = running_reward * self.alpha + ep_reward * (1-self.alpha)

            self.finish_episode(self.optimizer, rewards, self.policy)



    def finish_episode(self, rewards):
        R = 0
        DISCOUNT_FACTOR = 0.99
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r+DISCOUNT_FACTOR*R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std())
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]




class TemporalDifferenceReinforceTrainer(Reinforce):

    def train(self):



        for i_episode in range(1):
            seq = generate_random_sequence(self.N)
            env.reset(seq)
            state = env.rna.structure_representation
            ep_reward = 0

            for t in range(self. MAX_ITER):
                rewards = []
                action =  self.select_action(convert_to_tensor(state,seq),self.policy)
                state, reward, done, _ = env.step(action,self.N)
                rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            running_reward = running_reward * self.alpha + ep_reward * (1-self.alpha)

            self.finish_episode(self.optimizer, rewards, self.policy)


    def update_policy(self,optimizer, reward ,policy):
        pass

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



def convert_to_tensor(list,sequence):
    n = len(sequence)
    tensor = torch.tensor(np.zeros((1,1,8,n)),dtype=torch.double)
    base_index = 0
    for base in sequence:
        position = 0
        if base == 'A': position = 0
        elif base == 'U': position = 2
        elif base == 'G' : position = 4
        else : position = 6
        if len([ (x,y) for x, y in list if x  == base_index or y == base_index ]) != 0:
            tensor[0][0][position+1][base_index] = 1
        else:
            tensor[0][0][position][base_index] = 1
        base_index +=1

    return tensor



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




