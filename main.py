import train
import rna

def main():
    r = rna.RNA(train.generate_random_sequence(10))
    print(r)
    r.pairing(0,1)
    print(r)
    r.pairing(0, 1)
    r.pairing(1, 2)
    r.pairing(0, 2)
    r.pairing(0, 5)
    r.pairing(3, 4)
    trainer1 =train.MonteCarloReinforceTrainer()
    trainer1.train()


if __name__ == "__main__":
    main()

