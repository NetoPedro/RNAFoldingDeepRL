import train
import rna
import numpy as np
import predictor
import subprocess
import pandas as pd
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
    #mlp.show(arc_diagram.arc_diagram(arc_diagram.phrantheses_to_pairing_list(r.structure_representation_dot),sequence=r.sequence))

    diagram_dir = "./Predictions/"
    weights_dir = "./Weights/"

    stats = pd.DataFrame()
    stats["Episodes"] = [100,200,500,1000,2000]
    stats["Exploration Eps"] = [20,50,100,200,400]

    trainer1 =train.MonteCarloReinforceTrainer()
    trainer1.number_episodes = 100
    trainer1.exploration_eps = 20
    trainer1.train()


    predictor_reinforce = predictor.ReinforcePredictor()
    predictor_reinforce.predict("AAAACCUUUU",weights_dir+"monte_carlo_reinforce"+str(trainer1.number_episodes),diagram_dir+ "AAAACCUUUU" + "monte_carlo_reinforce" + str(trainer1.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir+"monte_carlo_reinforce"+str(trainer1.number_episodes),diagram_dir+"AGACCGGUCU" + "monte_carlo_reinforce" + str(trainer1.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir+"monte_carlo_reinforce" + str(trainer1.number_episodes),
                        diagram_dir+"CGUACGUACG" + "monte_carlo_reinforce" + str(trainer1.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU",weights_dir+ "monte_carlo_reinforce" + str(trainer1.number_episodes),
                        diagram_dir+"CGCCCCAAAU" + "monte_carlo_reinforce" + str(trainer1.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA", weights_dir+"monte_carlo_reinforce" + str(trainer1.number_episodes),
                        diagram_dir+"AUGCUGAUGA" + "monte_carlo_reinforce" + str(trainer1.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir+"monte_carlo_reinforce" + str(trainer1.number_episodes),
                        diagram_dir+"AAAUUUGGGC" + "monte_carlo_reinforce" + str(trainer1.number_episodes))

    trainer2 = train.MonteCarloReinforceTrainer()
    trainer2.number_episodes= 200
    trainer2.exploration_eps = 50
    trainer2.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir+"monte_carlo_reinforce" + str(trainer2.number_episodes),
                        diagram_dir+"AAAACCUUUU" + "monte_carlo_reinforce" + str(trainer2.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir+"monte_carlo_reinforce" + str(trainer2.number_episodes),
                        diagram_dir+"AGACCGGUCU" + "monte_carlo_reinforce" + str(trainer2.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir+"monte_carlo_reinforce" + str(trainer2.number_episodes),
                        diagram_dir+"CGUACGUACG" + "monte_carlo_reinforce" +str(trainer2.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir+"monte_carlo_reinforce" + str(trainer2.number_episodes),
                        diagram_dir+"CGCCCCAAAU" + "monte_carlo_reinforce" + str(trainer2.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA", weights_dir+"monte_carlo_reinforce" + str(trainer2.number_episodes),
                        diagram_dir+"AUGCUGAUGA" + "monte_carlo_reinforce" + str(trainer2.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir+"monte_carlo_reinforce" + str(trainer2.number_episodes),
                        diagram_dir+"AAAUUUGGGC" + "monte_carlo_reinforce" + str(trainer2.number_episodes))

    trainer3 = train.MonteCarloReinforceTrainer()
    trainer3.number_episodes = 500
    trainer3.exploration_eps = 100
    trainer3.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir+"monte_carlo_reinforce" + str(trainer3.number_episodes),
                        diagram_dir+"AAAACCUUUU" + "monte_carlo_reinforce" + str(trainer3.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU",weights_dir+ "monte_carlo_reinforce" + str(trainer3.number_episodes),
                        diagram_dir+"AGACCGGUCU" + "monte_carlo_reinforce" + str(trainer3.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir+"monte_carlo_reinforce" + str(trainer3.number_episodes),
                        diagram_dir+"CGUACGUACG" + "monte_carlo_reinforce" + str(trainer3.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir+"monte_carlo_reinforce" + str(trainer3.number_episodes),
                        diagram_dir+"CGCCCCAAAU" + "monte_carlo_reinforce" + str(trainer3.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA",weights_dir+ "monte_carlo_reinforce" + str(trainer3.number_episodes),
                        diagram_dir+"AUGCUGAUGA" + "monte_carlo_reinforce" + str(trainer3.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir+"monte_carlo_reinforce" + str(trainer3.number_episodes),
                        diagram_dir+"AAAUUUGGGC" + "monte_carlo_reinforce" + str(trainer3.number_episodes))

    trainer4 = train.MonteCarloReinforceTrainer()
    trainer4.number_episodes = 1000
    trainer4.exploration_eps = 200
    trainer4.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir+ "monte_carlo_reinforce" + str(trainer4.number_episodes),
                        diagram_dir+"AAAACCUUUU" + "monte_carlo_reinforce" + str(trainer4.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir+"monte_carlo_reinforce" + str(trainer4.number_episodes),
                        diagram_dir+"AGACCGGUCU" + "monte_carlo_reinforce" + str(trainer4.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir+"monte_carlo_reinforce" + str(trainer4.number_episodes),
                        diagram_dir+"CGUACGUACG" + "monte_carlo_reinforce" + str(trainer4.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir+"monte_carlo_reinforce" + str(trainer4.number_episodes),
                        diagram_dir+"CGCCCCAAAU" + "monte_carlo_reinforce" + str(trainer4.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA", weights_dir+"monte_carlo_reinforce" + str(trainer4.number_episodes),
                        diagram_dir+"AUGCUGAUGA" + "monte_carlo_reinforce" + str(trainer4.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir+"monte_carlo_reinforce" + str(trainer4.number_episodes),
                        diagram_dir+"AAAUUUGGGC" + "monte_carlo_reinforce" + str(trainer4.number_episodes))


    trainer5 = train.MonteCarloReinforceTrainer()
    trainer5.number_episodes = 2000
    trainer5.exploration_eps = 400
    trainer5.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir+"monte_carlo_reinforce" + str(trainer5.number_episodes),
                        diagram_dir+"AAAACCUUUU" + "monte_carlo_reinforce" + str(trainer5.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir+"monte_carlo_reinforce" + str(trainer5.number_episodes),
                        diagram_dir+"AGACCGGUCU" + "monte_carlo_reinforce" + str(trainer5.number_episodes))

    predictor_reinforce.predict("CGUACGUACG",weights_dir+ "monte_carlo_reinforce" + str(trainer5.number_episodes),
                        diagram_dir+"CGUACGUACG" + "monte_carlo_reinforce" + str(trainer5.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir+"monte_carlo_reinforce" + str(trainer5.number_episodes),
                        diagram_dir+"CGCCCCAAAU" + "monte_carlo_reinforce" + str(trainer5.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA",weights_dir+ "monte_carlo_reinforce" + str(trainer5.number_episodes),
                        diagram_dir+"AUGCUGAUGA" + "monte_carlo_reinforce" + str(trainer5.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir+"monte_carlo_reinforce" + str(trainer5.number_episodes),
                        diagram_dir+"AAAUUUGGGC" + "monte_carlo_reinforce" + str(trainer5.number_episodes))



    stats["Reinforce MC Accuracy"] = [ str(trainer1.correct_predictions / (trainer1.number_episodes - trainer1.exploration_eps)*100) + "%",
                              str(trainer2.correct_predictions / (trainer2.number_episodes - trainer2.exploration_eps)*100)+"%",
                              str(trainer3.correct_predictions / (trainer3.number_episodes - trainer3.exploration_eps)*100)+"%",
                              str(trainer4.correct_predictions / (trainer4.number_episodes - trainer4.exploration_eps)*100)+"%",
                              str(trainer5.correct_predictions / (trainer5.number_episodes - trainer5.exploration_eps)*100)+"%"]

    stats["Reinforce MC Average Iteration"] = [trainer1.sum_iterations_done / (trainer1.number_episodes - trainer1.exploration_eps),
                             trainer2.sum_iterations_done / (trainer2.number_episodes - trainer2.exploration_eps),
                             trainer3.sum_iterations_done / (trainer3.number_episodes - trainer3.exploration_eps),
                             trainer4.sum_iterations_done / (trainer4.number_episodes - trainer4.exploration_eps),
                             trainer5.sum_iterations_done / (trainer5.number_episodes - trainer5.exploration_eps)]




##### TD Reinforce

    trainer1 =train.TemporalDifferenceReinforceTrainer()
    trainer1.number_episodes = 100
    trainer1.exploration_eps = 20
    trainer1.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir + "td_reinforce" + str(trainer1.number_episodes),
                        diagram_dir + "AAAACCUUUU" + "td_reinforce" + str(trainer1.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir + "td_reinforce" + str(trainer1.number_episodes),
                        diagram_dir + "AGACCGGUCU" + "td_reinforce" + str(trainer1.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir + "td_reinforce" + str(trainer1.number_episodes),
                        diagram_dir + "CGUACGUACG" + "td_reinforce" + str(trainer1.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir + "td_reinforce" + str(trainer1.number_episodes),
                        diagram_dir + "CGCCCCAAAU" + "td_reinforce" + str(trainer1.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA", weights_dir + "td_reinforce" + str(trainer1.number_episodes),
                        diagram_dir + "AUGCUGAUGA" + "td_reinforce" + str(trainer1.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir + "td_reinforce" + str(trainer1.number_episodes),
                        diagram_dir + "AAAUUUGGGC" + "td_reinforce" + str(trainer1.number_episodes))

    trainer2 = train.TemporalDifferenceReinforceTrainer()
    trainer2.number_episodes = 200
    trainer2.exploration_eps = 50
    trainer2.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir + "td_reinforce" + str(trainer2.number_episodes),
                        diagram_dir + "AAAACCUUUU" + "td_reinforce" + str(trainer2.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir + "td_reinforce" + str(trainer2.number_episodes),
                        diagram_dir + "AGACCGGUCU" + "td_reinforce" + str(trainer2.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir + "td_reinforce" + str(trainer2.number_episodes),
                        diagram_dir + "CGUACGUACG" + "td_reinforce" + str(trainer2.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir + "td_reinforce" + str(trainer2.number_episodes),
                        diagram_dir + "CGCCCCAAAU" + "td_reinforce" + str(trainer2.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA", weights_dir + "td_reinforce" + str(trainer2.number_episodes),
                        diagram_dir + "AUGCUGAUGA" + "td_reinforce" + str(trainer2.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir + "td_reinforce" + str(trainer2.number_episodes),
                        diagram_dir + "AAAUUUGGGC" + "td_reinforce" + str(trainer2.number_episodes))

    trainer3 = train.TemporalDifferenceReinforceTrainer()
    trainer3.number_episodes = 500
    trainer3.exploration_eps = 100
    trainer3.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir + "td_reinforce" + str(trainer3.number_episodes),
                        diagram_dir + "AAAACCUUUU" + "td_reinforce" + str(trainer3.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir + "td_reinforce" + str(trainer3.number_episodes),
                        diagram_dir + "AGACCGGUCU" + "td_reinforce" + str(trainer3.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir + "td_reinforce" + str(trainer3.number_episodes),
                        diagram_dir + "CGUACGUACG" + "td_reinforce" + str(trainer3.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir + "td_reinforce" + str(trainer3.number_episodes),
                        diagram_dir + "CGCCCCAAAU" + "td_reinforce" + str(trainer3.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA", weights_dir + "td_reinforce" + str(trainer3.number_episodes),
                        diagram_dir + "AUGCUGAUGA" + "td_reinforce" + str(trainer3.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir + "td_reinforce" + str(trainer3.number_episodes),
                        diagram_dir + "AAAUUUGGGC" + "td_reinforce" + str(trainer3.number_episodes))

    trainer4 = train.TemporalDifferenceReinforceTrainer()
    trainer4.number_episodes = 1000
    trainer4.exploration_eps = 200
    trainer4.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir + "td_reinforce" + str(trainer4.number_episodes),
                        diagram_dir + "AAAACCUUUU" + "td_reinforce" + str(trainer4.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir + "td_reinforce" + str(trainer4.number_episodes),
                        diagram_dir + "AGACCGGUCU" + "td_reinforce" + str(trainer4.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir + "td_reinforce" + str(trainer4.number_episodes),
                        diagram_dir + "CGUACGUACG" + "td_reinforce" + str(trainer4.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir + "td_reinforce" + str(trainer4.number_episodes),
                        diagram_dir + "CGCCCCAAAU" + "td_reinforce" + str(trainer4.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA", weights_dir + "td_reinforce" + str(trainer4.number_episodes),
                        diagram_dir + "AUGCUGAUGA" + "td_reinforce" + str(trainer4.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir + "td_reinforce" + str(trainer4.number_episodes),
                        diagram_dir + "AAAUUUGGGC" + "td_reinforce" + str(trainer4.number_episodes))

    trainer5 = train.TemporalDifferenceReinforceTrainer()
    trainer5.number_episodes = 2000
    trainer5.exploration_eps = 400
    trainer5.train()

    predictor_reinforce.predict("AAAACCUUUU", weights_dir + "td_reinforce" + str(trainer5.number_episodes),
                        diagram_dir + "AAAACCUUUU" + "td_reinforce" + str(trainer5.number_episodes))
    predictor_reinforce.predict("AGACCGGUCU", weights_dir + "td_reinforce" + str(trainer5.number_episodes),
                        diagram_dir + "AGACCGGUCU" + "td_reinforce" + str(trainer5.number_episodes))

    predictor_reinforce.predict("CGUACGUACG", weights_dir + "td_reinforce" + str(trainer5.number_episodes),
                        diagram_dir + "CGUACGUACG" + "td_reinforce" + str(trainer5.number_episodes))
    predictor_reinforce.predict("CGCCCCAAAU", weights_dir + "td_reinforce" + str(trainer5.number_episodes),
                        diagram_dir + "CGCCCCAAAU" + "td_reinforce" + str(trainer5.number_episodes))

    predictor_reinforce.predict("AUGCUGAUGA", weights_dir + "td_reinforce" + str(trainer5.number_episodes),
                        diagram_dir + "AUGCUGAUGA" + "td_reinforce" + str(trainer5.number_episodes))

    predictor_reinforce.predict("AAAUUUGGGC", weights_dir + "td_reinforce" + str(trainer5.number_episodes),
                        diagram_dir + "AAAUUUGGGC" + "td_reinforce" + str(trainer5.number_episodes))


    stats["Reinforce TD Accuracy"] = [ str(trainer1.correct_predictions / (trainer1.number_episodes - trainer1.exploration_eps)*100) + "%",
                              str(trainer2.correct_predictions / (trainer2.number_episodes - trainer2.exploration_eps)*100)+"%",
                              str(trainer3.correct_predictions / (trainer3.number_episodes - trainer3.exploration_eps)*100)+"%",
                              str(trainer4.correct_predictions / (trainer4.number_episodes - trainer4.exploration_eps)*100)+"%",
                              str(trainer5.correct_predictions / (trainer5.number_episodes - trainer5.exploration_eps)*100)+"%"]

    stats["Reinforce TD Average Iteration"] = [
        trainer1.sum_iterations_done / (trainer1.number_episodes - trainer1.exploration_eps),
        trainer2.sum_iterations_done / (trainer2.number_episodes - trainer2.exploration_eps),
        trainer3.sum_iterations_done / (trainer3.number_episodes - trainer3.exploration_eps),
        trainer4.sum_iterations_done / (trainer4.number_episodes - trainer4.exploration_eps),
        trainer5.sum_iterations_done / (trainer5.number_episodes - trainer5.exploration_eps)]



    stats.to_html('stats.html')
    subprocess.call(
        'wkhtmltoimage -f png --width 0 stats.html stats.png', shell=True)

    #predictorTD = predictor.ReinforcePredictor()
    #predictorTD.predict("AAAACCUUUU","td_reinforce")
    #predictorTD.predict("AGACCGGUCU", "td_reinforce")





if __name__ == "__main__":
    main()

