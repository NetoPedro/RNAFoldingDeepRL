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
  
  
#### Pairing Function 

### Environment

The environment is responsible for storing a RNA object and return rewards based on an action. 

#### Reset Function
 
#### Step Function
  
  ##### Rewards
  



### Policy

  #### Deep Convolutional Neural Network
