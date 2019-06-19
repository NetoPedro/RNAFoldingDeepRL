# RNAFoldingDeepRL

Project for the course Machine Learning in Bioinformatics, at Aalto University.

  ![Sakurambo [CC BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0/)]](https://upload.wikimedia.org/wikipedia/commons/3/3f/Stem-loop.svg)

Sakurambo [CC BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0/)]

## Overview

This project aims to tackle a simplified version of the RNA folding (secondary structure) problem using reinforcement learning. To achieve the mentionated goal it will be necessary to build the entire environment from scratch, including rewards, states and other necessary utilities.  

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
  
  The structure representation is not unique, in a way that the secondary structure of a RNA sequence can be represented by a myriad of ways. Some representations are better to detect pseudoknots, others are better to feed to a policy. 
  
  ![Structure Example](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/RNAStructure.png)
   
   Above it is possible to see a common representation of the secondary structure of a RNA sequence. Below it is possible to see a ar diagram representation of other sequence.
   
   ![Structure Example](https://raw.githubusercontent.com/NetoPedro/RNAFoldingDeepRL/master/arc_diagram.png)
  
#### Implementation of the structure 

The folding proposed is constrained by the following rules:
- Pseudoknot or crossing are not allowed
- A certain base must belong to at most one pair • A-U pairing is permitted
- C-G pairing is permitted

The structure has a myriad of representations, although for the purpose of this project only three representations are going to be considered. The first, attending to the rules above, is a dot matrix. This is a very good visualization approach, although verifying crosses is expensive. To second representation is a pairing list, a much simpler approach to verify crosses and pseudoknots. Finally the last representation is an One-Hot Encoded matrix detailed on the Policy subsection below.

### State 

The state represents the system at a given step. In this project, the state is defined as the RNA structure at that given step. For example, when the system is initialized the RNA structure does not have any pair, at each step a pairing function tries to update the state to the next step. The state has all the information needed to act on the system.

### Action

Actions are applied over the current state of the system generating another state (sometimes the state remains equal). An action in this problem is characterized by and attempt to pair/unpair two given bases together. It is possible that an action is not possible to be performed, resulting on a similar state.

### Environment

The state of the system, and all the necessary functions to perform a step are packed in the environment. It is responsible to initialize all the necessary components to the system and is one of the most crucial components. Every step call performs an action, generating a new state, a reward value and an indication if it is possible to further improve from that step to the next ones. It should also include a reset function to reestablish the state of the system.
  
### Rewards
  
When an action is performed over a state, the action generates a possible change on the state. From this change it will result a value representing how good was the action performed. The value is usually known as the reward given by some action. Rewards can be positive, negative or just 0 (for example if the state does not change). Regarding this problem, the reward is the difference of connected bases on the state<sub>t</sub> and state<sub>t+1</sub>.


### Policy

Exploitation is based on the current knowledge about the problem. To generate actions it is necessary to have a representation of the current knowledge. This representation is usually a mathematical function that predicts what should be done next, given a state as input. At the beginning the knowledge represented by the policy is much similar to a random exploration, but during the training process given some rewards the function parameters are updated to approximate the function to the optimal representation of the system dynamics.

The policy can be any mathematical technique able to represent a complex enough representation of the dynamics present on the system. In this paper, a deep convolutional neural network represents the policy as seen on the figure below. These networks were popularized by image recognition and other image-related tasks. Nevertheless, they perform generally good with most matrix-based inputs.

![CNN Example](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

Aphex34 [CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0)]

As the input for the network, the state is represented by an 8*n matrix, where n stands for the size of the sequence. This representation simulates for each position from 0 to n and hot-encoding on the following categories:
1. Base A unpaired
2. Base A paired
3. Base U unpaired 
4. Base U paired
5. Base C unpaired 
6. Base C paired
7. Base G unpaired 
8. Base G paired
 




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


