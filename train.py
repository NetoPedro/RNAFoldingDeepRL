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
    MAX_ITER = 1000
    exploration_eps = 10
    N = 10
    alpha = .90
    correct_predictions = 0
    sum_iterations_done = 0
    number_episodes = 100;
    policy = policy_module.Policy(N)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-6)
    weights_dir = "./Weights/"
    def select_action(self,state, policy):
        probs = policy.forward(state)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_probs.append(m.log_prob(action))
        item = action.item()
        return (int(item/self.N),item%self.N)

class MonteCarloReinforceTrainer(Reinforce):

    def train(self):


        for i_episode in range(self.number_episodes):
            if i_episode%10 == 0 :
                print("Episode : ", i_episode)
            seq = generate_random_sequence(self.N)
            env.reset(seq)
            state = env.rna.structure_representation
            ep_reward = 0
            rewards = []
            bestState = env.rna.structure_representation_dot.copy()
            bestReward = 0
            for t in range(self.MAX_ITER):
                exploration = np.random.uniform(0,1)
                if exploration > 0.3 or i_episode > self.exploration_eps or t == 0:#.9 * 1/(i_episode+1):
                    action = self.select_action(convert_to_tensor(state, seq), self.policy)
                    state, reward, done, _ = env.step(action,self.N)
                    rewards.append(reward)
                else:
                    action = np.random.randint(0,self.N,2)
                    action = (action[0],action[1])
                    state, reward, done, _ = env.step(action, self.N)
                    rewards.append(reward)
                ep_reward += reward
                if ep_reward > bestReward:
                    bestState = env.rna.structure_representation_dot.copy()
                    bestReward = ep_reward

                if done:
                    if i_episode > self.exploration_eps:
                        self.correct_predictions +=1
                        self.sum_iterations_done += t
                    print("Done ", i_episode, " iteration ", t)
                    title = str(i_episode) + " Done at iteration " + str(t)
                    if i_episode >= 2090:
                        mlp.show(arc_diagram.arc_diagram(
                            arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot),seq, title))

                    break
                if (t+1)%100 == 0 and i_episode >= 2090:
                   mlp.show(arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot),seq,i_episode))

            self.running_reward = self.running_reward * self.alpha + ep_reward * (1-self.alpha)
            if i_episode >= 2000:
                mlp.show(
                    arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(bestState),
                                            seq, "Best State Achieved"))

            self.finish_episode(rewards)
        self.policy.save_weights(self.weights_dir+ "monte_carlo_reinforce"+str(self.number_episodes))


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


        for i_episode in range(self.number_episodes):
            seq = generate_random_sequence(self.N)
            env.reset(seq)
            state = env.rna.structure_representation
            ep_reward = 0
            rewards = []
            bestState = env.rna.structure_representation_dot.copy()
            bestReward = 0
            for t in range(self.MAX_ITER):
                exploration = np.random.uniform(0,1)
                if exploration > 0.3 or i_episode > 100 or t == 0:
                    action = self.select_action(convert_to_tensor(state, seq), self.policy)
                    state, reward, done, _ = env.step(action,self.N)
                    rewards.append(reward)
                    self.update_policy(reward)
                else:
                    action = np.random.randint(0,self.N,2)
                    action = (action[0],action[1])
                    state, reward, done, _ = env.step(action, self.N)
                    rewards.append(reward)

                ep_reward += reward
                if ep_reward > bestReward:
                    bestState = env.rna.structure_representation_dot.copy()
                    bestReward = ep_reward

                if done:
                    if i_episode > self.exploration_eps:
                        self.correct_predictions +=1
                        self.sum_iterations_done += t
                    print("Done ", i_episode, " iteration ", t)
                    title = str(i_episode) + " Done at iteration " + str(t)
                    if i_episode >= 3000:
                        mlp.show(arc_diagram.arc_diagram(
                            arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot), seq, title))

                    break

                if (t + 1) % 100 == 0 and i_episode >= 3000:
                    mlp.show(arc_diagram.arc_diagram(
                        arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot), seq, i_episode))

                self.running_reward = self.running_reward * self.alpha + ep_reward * (1 - self.alpha)
            if i_episode >= 3000:
                mlp.show(
                    arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(bestState),
                                            seq, "Best State Achieved"))

            self.finish_episode()
        self.policy.save_weights(self.weights_dir+"td_reinforce"+str(self.number_episodes))

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




