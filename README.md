# Multi-Agent Reinforcement Learning Competitive Environment

## Overview  
This repository implements a custom **multi-agent reinforcement learning (MARL)** environment with a competitive setting. It is intended as a testbed for training RL agents — possibly with self-play or adversarial setups — where multiple agents compete in a shared environment.  

The code is written in Python and provides a basic environment definition, an agent interface, and a training loop (in `main.py` / `rl.py`), to allow experimenting with different RL algorithms in a multi-agent competitive context.

Original environment: [Link Text](https://github.com/KRLGroup/gym-ma-survival-2d/tree/main)


## Features  
- A minimal multi-agent environment scaffold supporting competition between agents.  
- Simple agent abstraction defined in `agent.py` — easy to extend/customize for different agent types.  
- Example training / evaluation loop (e.g. via `main.py`, `rl.py`) to quickly launch experiments.  
- Lightweight and easy to understand — ideal as a starting point for further research or as a learning tool for MARL.  


## Repository Structure  
- **env/**: Contains the environment implementation (state, observations, step/reset logic).  
- **agent.py**: Defines how agents observe the environment, choose actions, and (optionally) learn/update policies.  
- **rl.py**: Implements the logic to run episodes, collect experiences, compute rewards, and loop training / evaluation.  
- **main.py**: Example entry point — shows how to instantiate the environment and agents, how to run training, evaluation, or simulation.  


## License & Credits  
Feel free to reuse or adapt this code for educational, research or personal purposes.
