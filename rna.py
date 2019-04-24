import numpy as np
SEQUENCE_LENGTH = 10


class RNA:
    def __init__(self):
        self.sequence = ""  # TODO Define the initial sequence
        self.representation =  np.zeros() # TODO Find a proper representation (dot matrix?)
        self.free_energy = SEQUENCE_LENGTH   # For now this represents the number of elements in the sequence - connections
    def pairing(self, nucleo1, nucleo2):
        pass

