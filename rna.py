import numpy as np
SEQUENCE_LENGTH = 10


class RNA:
    def __init__(self, sequence):
        self.sequence = sequence # TODO Define the initial sequence
        self.representation =  self.build_representations(self.sequence)
        self.free_energy = self.update_free_energy()


    def pairing(self, nucleo1, nucleo2):
        pass


    def build_representations(self,sequence):
        # TODO Find a proper representation (dot matrix?)
        return sequence

    #  This function is here for decoupling in case a new and more accurate energy function is given.
    def update_free_energy(self,representation):
        # For now this represents the number of elements in the sequence - connections
        return 0

