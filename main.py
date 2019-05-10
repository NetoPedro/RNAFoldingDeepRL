import train
import rna
import arc_diagram
import matplotlib.pyplot as mlp
def main():
    r = rna.RNA(train.generate_random_sequence(20))
    print(r)
    r.pairing(1,2)
    print(r)
    #r.pairing(0, 1)
    r.pairing(2, 5)
    r.pairing(3, 4)
    r.pairing(6, 9)
    r.pairing(0, 19)

    r.pairing(7, 8)
    r.pairing(10, 18)
    r.pairing(11, 17)
    r.pairing(12, 16)
    r.pairing(13, 14)
    mlp.show(arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(r.structure_representation_dot),sequence=r.sequence))
    trainer1 =train.MonteCarloReinforceTrainer()
    trainer1.train()


if __name__ == "__main__":
    main()

