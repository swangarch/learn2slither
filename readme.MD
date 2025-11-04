# Learn2Slither

A **reinforcement learning** project that trains an AI agent to play the classic Snake game using a **Deep Q-Network (DQN)** implemented **from scratch**.

The project is still under optimization

## Overview

**Learn2Slither** uses a Deep Q-Network based on a custom-built **multi-layer perceptron (MLP)** written entirely with **NumPy** — without using deep learning frameworks.  

The agent learns to maximize its snake length through reinforcement learning, improving its policy via experience replay and Q-learning updates.

## Features

- Custom Snake environment for training and testing  
- Fully manual implementation of a **Deep Q-Network (DQN)**  
- Core neural network (MLP) built from scratch using **NumPy only**  
- Replay memory and epsilon-greedy exploration  
- Visualization of training progress and gameplay  

## Technologies Used

- Python  
- NumPy  
- Matplotlib (for training visualization)  
- Pygame (for visualization of snake)  

## Getting Started

### Setup

Clone this repository:
```bash
git clone https://github.com/swangarch/learn2slither.git
cd learn2slither

make setup
source venv/bin/activate # to setup virtual enviroment
```

```bash
make start # to train the model
```
During the training, you can press Q to slow down the training progress to observe, and E to resume.
