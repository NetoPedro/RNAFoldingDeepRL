# RNAFoldingDeepRL

Project for the course Machine Learning in Bioinformatics, at Aalto University.

  ![Sakurambo [CC BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0/)]](https://upload.wikimedia.org/wikipedia/commons/3/3f/Stem-loop.svg)

Sakurambo [CC BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0/)]

## Overview

This project aims to tackle a simplified version of the RNA folding (secondary structure) problem using reinforcement learning. To achieve the mentionated goal it will be necessary to build the entire environment from scratch, including rewards, states and other necessary utilities. The project will be structured in 3 main components. 

* RNA
* Environment 
* Policy

All those components are further explained below. 

## Installation Dependencies:
* Python 2.7 or 3
* Pytorch

## How to Run?
```
git clone https://github.com/NetoPedro/RNAFoldingDeepRL.git
cd RNAFoldingDeepRL
python main.py
```

## Components

### RNA 

The RNA component has the main intention to model the behaviour of a RNA sequence and respective structure. This model is created in a simplified way.

 
#### Sequence

  The sequence is a string of the characters "A", "C", "U", "G" that represent the bases present on the RNA. 
  
  ![Sequence Example](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/RNASequence.png)

#### Structure Representation 
  
  The structure representation is not unique, in a way that the secondary structure of a RNA sequence can be represented by a myriad of ways. Some representations are better to detect pseudoknots, others are better to feed to a policy, therefore there are 2 different representations in this project. 
  
  ![Structure Example](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/RNAStructure.png)
   
   Above it is possible to see a common representation of the secondary structure of a RNA sequence. Below it is possible to see a ar diagram representation of other sequence.
   
   ![Structure Example](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/arc_diagram.png)
  
  
#### Free Energy
  
  The free energy is calculated as the number of bases not connected to another. Since we are working on a simplified version of the problem, the only type of connections is a single connection between a pair of bases, where each base can only belong to one pair. Also pseudoknots are not allowed. 
  The free energy is lower bounded by the left of the number of bases divided by 4. Further bounds can be set considering that not all bases can be connected to each other. 
  
#### Pairing Function 
 
 This function is responsible for connecting two bases. Because of the restrictions set above, this function needs to verify some flaws in the given data. The function has 2 inputs, the position of each base. In order to connected them it is important to first verify if the bases are possible to be connected, checking for A-U and C-G pairs. Secondly, the connection must not be done if there is any other previous connection that originates a cross with the new one. Finally if the connection already exists it should be removed, otherwise added. 
  This function ends with an update to the energy function on every situation where the structure is changed. 

### Environment

The environment is responsible for storing a RNA object and return rewards based on an action.

#### Reset Function
 
 The reset function is the initial step at each epoch, to reset to default values the fields and variables of the environment. A new RNA object is also created to store the new sequence and with reseted values. This function is called by the __ init __ function.
 
#### Step Function

 The step function receives an action, and using that action it decomposes the action in 2 positions to be connected. This is followed by a call to the pairing function. The reward is constructed with the new value of free-energy subtracted to the old one. This let's us know if there was any kind of improvement to the structure. Finally, it is also responsible for checking if it is possible to further improve the model. If it isn't then the returned value 'done' must be set to true. 
  
##### Rewards
  



### Policy

#### Reinforce 

#### Actor Critic (A2C or A3C)

#### Deep Convolutional Neural Network
 TODO Talk about architecture, training, loss etc 
 
![CNN Example](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

Aphex34 [CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0)]



### Training

#### Monte Carlo 


#### TD(0) - Temporal Difference 


#### TD( \(\lambda\) )

### Results and Comparison 

A prediction is considered correct if its reward matches the upperbound. Nevertheless there are maximal foldings that are lower than the upperbound. Because of this detail the following results must be considered as approximations with some error margin. 


![Stats](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/stats_mc_td_reinforce_complete.png)

![monte_carlo1](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/Predictions/AGACCGGUCUmonte_carlo_reinforce2000.png)
![td1](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/Predictions/AGACCGGUCUtd_reinforce500.png)

Above it is possible to see 2 predicted structures. The first is done by a monte carlo reinforce method after 2000 epochs. 
The second is done by a td(0) reinforce after 500 epochs.

![monte_carlo12](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/Predictions/AUGCUGAUGAmonte_carlo_reinforce2000.png)
![td2](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/Predictions/AUGCUGAUGAtd_reinforce1000.png)

Above it is possible to see 2 predicted structures. The first is done by a monte carlo reinforce method after 2000 epochs. 
The second is done by a td(0) reinforce after 1000 epochs.


