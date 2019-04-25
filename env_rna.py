import rna as rna_module

class EnvRNA:
    def __init__(self,sequence = ""):
        self.reset(sequence)

    def reset(self,sequence):
        self.rna = rna_module.RNA(sequence)

    def step(self,action,N = 10):
        previous_value = self.rna.free_energy
        # TODO Call RNA pairing to update accordingly with the action
        new_value = 0
        reward = new_value - previous_value
        done = False
        if N == new_value:
            done = True
        return self.rna.structure_representation,reward,done,""  # This values represent Observation, Reward, Done and info
