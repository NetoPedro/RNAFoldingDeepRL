import numpy as np
class RNA:

    def __init__(self, sequence):
        self.sequence = sequence
        self.structure_representation,self.structure_representation_dot =  self.build_representations(self.sequence)

        self.free_energy = self.update_free_energy()


    def pairing(self, nucleo1, nucleo2):
        # TODO Verify bases before connecting them. A-U C-G
        if not(self.sequence[nucleo1] == "A" and self.sequence[nucleo2] == "U"):
            if not(self.sequence[nucleo1] == "C" and self.sequence[nucleo2] == "G"):
                return
        if nucleo1 == nucleo2: return
        if nucleo1 > nucleo2:
            nucleo1, nucleo2 = nucleo2, nucleo1
        #if nucleo2 - nucleo1 <= 3:
        #    return
        if len(self.structure_representation) > 0 :
            for (n1,n2) in self.structure_representation:
                if nucleo1 > n2 and nucleo2 > n2:
                    continue
                if nucleo1 < n1 and nucleo2 < n1:
                    continue
                if not((nucleo1 < n1 and nucleo2 > n2) or (nucleo1 > n1 and nucleo2 < n2) or (nucleo1 == n1 and nucleo2 == n2)):
                    return

        if (nucleo1,nucleo2) in self.structure_representation:
            self.structure_representation.remove((nucleo1, nucleo2))
            self.structure_representation_dot[nucleo1] = "."
            self.structure_representation_dot[nucleo2] = "."
        else:
            self.structure_representation.append((nucleo1, nucleo2))
            self.structure_representation_dot[nucleo1] = "("
            self.structure_representation_dot[nucleo2] = ")"

        self.free_energy = self.update_free_energy()

    def build_representations(self,sequence):
        return [],  list("".ljust(len(sequence),"."))

    #  This function is here for decoupling in case a new and more accurate energy function is given.
    def update_free_energy(self):
        # For now this represents the number of connections
        return 2 *len(self.structure_representation)

