import policy as policy_module
import env_rna
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as mlp
import arc_diagram
from torch.distributions import Categorical

env = env_rna.EnvRNA()

class Reinforce:


    running_reward = 0
    MAX_ITER = 300
    N = 10
    alpha = .95

    policy = policy_module.Policy(N)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    def select_action(self,state, policy):
        probs = policy.forward(state)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_probs.append(m.log_prob(action))
        item = action.item()
        return (int(item/self.N),item%self.N)

class MonteCarloReinforceTrainer(Reinforce):

    def train(self):


        for i_episode in range(500):
            seq = generate_random_sequence(self.N)
            env.reset(seq)
            state = env.rna.structure_representation
            ep_reward = 0
            rewards = []
            for t in range(self.MAX_ITER):
                exploration = np.random.uniform(0,1)
                if exploration > 0.1 or i_episode > 350:#.9 * 1/(i_episode+1):
                    action = self.select_action(convert_to_tensor(state, seq), self.policy)
                    state, reward, done, _ = env.step(action,self.N)
                    rewards.append(reward)
                else:
                    action = np.random.randint(0,self.N,2)
                    action = (action[0],action[1])
                    state, reward, done, _ = env.step(action, self.N)
                    rewards.append(reward)
                ep_reward += reward
                if done:
                    if i_episode >= 490:
                        mlp.show(arc_diagram.arc_diagram(
                            arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot),seq))
                    break
                if (t+1)%100 == 0 and i_episode >= 490:
                   mlp.show(arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot),seq))

            self.running_reward = self.running_reward * self.alpha + ep_reward * (1-self.alpha)

            self.finish_episode(rewards)



    def finish_episode(self, rewards):
        R = 0
        DISCOUNT_FACTOR = 0.99
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r+DISCOUNT_FACTOR*R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean())
        for log_prob, R in zip(self.policy.saved_probs, returns):
            loss = -log_prob * R
            policy_loss.append(loss.unsqueeze(0))
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.saved_probs[:]




class TemporalDifferenceReinforceTrainer(Reinforce):


    def train(self):


        for i_episode in range(100):
            seq = generate_random_sequence(self.N)
            env.reset(seq)
            state = env.rna.structure_representation
            ep_reward = 0
            rewards = []
            for t in range(self.MAX_ITER):
                exploration = 1#np.random.uniform(0,1)
                if exploration > .9 * 1/(i_episode+1):
                    action = self.select_action(convert_to_tensor(state, seq), self.policy)
                    state, reward, done, _ = env.step(action,self.N)
                    rewards.append(reward)
                else:
                    action = np.random.randint(0,self.N,2)
                    action = (action[0],action[1])
                    state, reward, done, _ = env.step(action, self.N)
                    rewards.append(reward)
                ep_reward += reward
                if done:
                    break
                if (t+1)%100 == 0 and i_episode >= 50:
                   mlp.show(arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot),seq))
                self.running_reward = self.running_reward + self.alpha * (reward + 0.9 * 2 - self.running_reward)
                self.update_policy(self.running_reward)



            self.finish_episode()


    def update_policy(self,reward):
        R = 0
        DISCOUNT_FACTOR = 0.99
        policy_loss = []
        returns = []
        R = reward
        loss = -self.policy.saved_probs[-1] * R
        policy_loss.append(loss.unsqueeze(0))
        self.optimizer.zero_grad()
        policy_loss = F.smooth_l1_loss(2,loss)
        policy_loss.backward()
        self.optimizer.step()


    def finish_episode(self):
        del self.policy.saved_probs[:]



def convert_to_tensor(list,sequence):
    #TODO Change to NxN tensor
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




