import rna as rna_module

class EnvRNA:
    def __init__(self,sequence = ""):
        self.reset(sequence)

    def reset(self,sequence):
        self.rna = rna_module.RNA(sequence)

    def step(self,action):

        return 0,0,0,0  # This values represent Observation, Reward, Done and info
