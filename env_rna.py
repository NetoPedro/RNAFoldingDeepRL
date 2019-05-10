import rna as rna_module

class EnvRNA:
    def __init__(self,sequence = ""):
        self.reset(sequence)


    def reset(self,sequence):
        self.rna = rna_module.RNA(sequence)
        self.maxValue = min(sequence.count("A"), sequence.count("U")) + min(sequence.count("C"), sequence.count("G"))

    def step(self,action_pair,N = 10):
        previous_value = self.rna.free_energy
        self.rna.pairing(action_pair[0],action_pair[1])
        new_value = self.rna.free_energy
        reward = new_value - previous_value
        done = False
        if 2*self.maxValue == new_value:
            done = True
        #if reward == 0: reward = -0.1
        return self.rna.structure_representation,reward,done,""  # This values represent Observation, Reward, Done and info
