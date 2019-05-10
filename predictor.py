import train
import env_rna
import matplotlib.pyplot as mlp
import arc_diagram
import torch
class ReinforcePredictor(train.Reinforce):

    def predict(self,seq):
            env = env_rna.EnvRNA()
            self.policy.load_weights("monte_carlo_reinforce")
            env.reset(seq)
            state = env.rna.structure_representation
            ep_reward = 0
            rewards = []
            bestState = env.rna.structure_representation_dot.copy()
            bestReward = 0
            for t in range(self.MAX_ITER):
                action = self.select_action(train.convert_to_tensor(state, seq), self.policy)
                state, reward, done, _ = env.step(action,self.N)
                rewards.append(reward)

                ep_reward += reward
                if ep_reward > bestReward:
                    bestState = env.rna.structure_representation_dot.copy()
                    bestReward = ep_reward

                if done:
                    print("Done iteration ", t)
                    title =  " Done at iteration " + str(t)
                    mlp.show(arc_diagram.arc_diagram(
                            arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot),seq, title))

                    break

            self.running_reward = self.running_reward * self.alpha + ep_reward * (1-self.alpha)
            mlp.show(
                arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(bestState),
                                        seq, "Best State Achieved"))






