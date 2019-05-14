import train
import env_rna
import matplotlib.pyplot as mlp
import arc_diagram
from torch.distributions import Categorical
class ReinforcePredictor(train.Reinforce):


    def select_action(self, state, policy):
        probs = policy.forward(state)
        m = Categorical(probs)
        action = m.sample()
        item = action.item()
        return (int(item / self.N), item % self.N)

    def predict(self,seq,weights_name,diagram_name):

            env = env_rna.EnvRNA()
            self.policy.load_weights(weights_name)
            env.reset(seq)
            finished = False
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
                    finished = True
                    title =  " Done at iteration " + str(t)
                    #mlp.show(arc_diagram.arc_diagram(
                    #        arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot),seq, title))
                    arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(env.rna.structure_representation_dot),
                                            seq, title)
                    mlp.savefig(diagram_name)
                    break

            self.running_reward = self.running_reward * self.alpha + ep_reward * (1-self.alpha)
            if not(finished):
                arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(bestState),
                                    seq, "Best State Achieved")
                mlp.savefig(diagram_name)






