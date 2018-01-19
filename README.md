# Recurrent-Environment-Simulators
An implementation of Deepmind recurrent enviroment simulators in tensorflow. 


<div align="center">
<img src="imgs/2.png"><br><br>
</div>

# Action-conditioned LSTM
The paper used a modified version of LSTM called Action conditioned LSTM, mainly it's an early fusion between actions and states. They used this approach as it enables them to explore how the model generalises to different action policies.


<div align="center">
<img src="imgs/1.png"><br><br>
</div>

## Data Collection
We trained a synchronous Advantage Actor Critic (A2C) agent and used it to collect data from openAi Atari enviroments.


## Usage
  #####Dependencies
```
Python 3.X
tensorflow 1.3.0
numpy 1.13.1
tqdm
```
  #####Train

  - Collect data from any atari enviroment using the method mentioned before, or use the provided data.
- Run ```python res.py is_train=True```


