# Recurrent Environment Simulators (RES)
Modeling the environment is an important task for intelligent agents to be able to plan and act efficiently. A Recurrent Environment Simulator network can achive this task easily by learning to predict the next observation given the history of observations and actions. Making the agent able to predict the consequences of its actions.

This repository contains a tensorflow implementation of the Recurrent Enviroment Simulators paper puplished by DeepMind at ICML 2017. (https://arxiv.org/abs/1704.02254)


# Action-conditioned LSTM
The paper used a modified version of LSTM called Action conditioned LSTM, mainly it's an early fusion between actions and states. They used this approach as it enables them to explore how the model generalises to different action policies. 

<div align="center">
<img src="imgs/1.png"><br><br>
</div>

## Data Collection
We trained a Synchronous Advantage Actor Critic (A2C) agent and used it to collect data from openAi Atari enviroments.


## Usage
  ##### Dependencies
```
Python 3.X
tensorflow 1.3.0
numpy 1.13.1
tqdm
```
  ##### Train

  - Collect data from any atari enviroment using the method mentioned before, or use the provided data.
- Run ```python res.py is_train=True```


