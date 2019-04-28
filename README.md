# RNAFoldingDeepRL

Project for the course Machine Learning in Bioinformatics, at Aalto University.

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

TODO Add explanation

 
#### Sequence

  The sequence is a string of the characters "A", "C", "U", "G" that represent the bases present on the RNA. 

#### Structure Representation 
  
  The structure representation is not unique, in a way that the secundary structure of a RNA sequence can be represented by a myriad of ways. Some representations are better to detect pseudoknots, others are better to feed to a policy, therefore there are 2 different representations in this project. 
  
#### Free Energy
  
  The free energy is calculated as the number of bases not connected to another. Since we are working on a simplified version of the problem, the only type of connections is a single connection between a pair of bases, where each base can only belong to one pair. Also pseudoknots are not allowed. 
  The free energy is lower bounded by the left of the number of bases divided by 4. Further bounds can be set considering that not all bases can be connected to each other. 
  
#### Pairing Function 
 
 This function is responsible for connecting two bases. Because of the restrictions set above, this function needs to verify some flaws in the given data. The function has 2 inputs, the position of each base. In order to connected them it is important to first verify if the bases are connectable, checking for A-U and C-G pairs. Secondly, the connection must not be done if there is any other previous connection that originates a cross with the new one. Finally if the connection already exists it should be removed, otherwise added. 
  This function ends with an update to the energy function on every situation where the structure is changed. 

### Environment

The environment is responsible for storing a RNA object and return rewards based on an action. 

#### Reset Function
 
#### Step Function
  
##### Rewards
  



### Policy

#### Deep Convolutional Neural Network
