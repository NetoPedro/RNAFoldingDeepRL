import numpy as np
SEQUENCE_LENGTH = 10


class RNA:
    def __init__(self, sequence):
        # if len(sequence) < SEQUENCE_LENGTH:
        #    padding_size = SEQUENCE_LENGTH - len(sequence)
        #    for i in range (padding_size):
        #        sequence += 'E'

        self.sequence = sequence # TODO Define the initial sequence
        self.structure_representation =  self.build_representations(self.sequence)
        self.free_energy = self.update_free_energy()


    def pairing(self, nucleo1, nucleo2):
        # TODO Verify bases before connecting them
        if len(self.structure_representation) > 0 :
            for (n1,n2) in self.structure_representation:
                if not((nucleo1 < n1 and nucleo2 > n2) or (nucleo1 > n1 and nucleo2 < n2) or (nucleo1 == n1 and nucleo2 == n2)):
                    return

        if (nucleo1,nucleo2) in self.structure_representation:
            self.structure_representation.remove((nucleo1, nucleo2))

        else:
            self.structure_representation.append((nucleo1, nucleo2))

        self.free_energy = self.update_free_energy()

    def build_representations(self,sequence):
        return []

    #  This function is here for decoupling in case a new and more accurate energy function is given.
    def update_free_energy(self):
        # For now this represents the number of elements in the sequence - connections
        return SEQUENCE_LENGTH - len(self.structure_representation)

