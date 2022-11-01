# Conditional RAT-SPNs for Reinforcement Learning

A PyTorch implementation of Random Tensorized Sum-Product Networks (RAT-SPNs) 
[1] with a conditional variant like in Conditional Sum-Product Networks [2]. 
This repository is the code for my Master's thesis that applies conditional SPNs to the 
soft actor-critic (SAC) reinforcement learning algorithm. 
Applying SPNs to SAC requires entropy approximations, of which four are implemented (the recursive one works best).
This repository is based on some files from [SPFlow](https://github.com/SPFlow/SPFlow). 

* The reinforcement learning experiments can be found under `experiments/joint_fail_sac_sb3.py` and evaluates 
SAC with an SPN policy on MuJoCo environments. 
The RL stuff is built on the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) framework.
* A simple experiment that simply maximizes the SPN entropy can be found in `experiments/entropy.py`. 
* An experiment that learns the distribution of MNIST is implemented under `experiments/mnist_gen_train.py`. 

Use this repository in a Python 3.8 environment. 
If you like to use Conda, you can create the environment from `conda_env.yml`. 

### References

[1] Peharz et al., 2018: Probabilistic Deep Learning using Random Sum-Product Networks

[2] Shao et al., 2022: Conditional sum-product networks: Modular probabilistic circuits via gate functions


