# Learn2Slither

A **reinforcement learning** project that trains an AI agent to play the classic **Snake** game using a **Deep Q-Network (DQN)** implemented from **scratch**.

The project is still under optimization.

## Overview

**Learn2Slither** uses a Deep Q-Network built on a custom **multilayer perceptron (MLP)** — without relying on any deep learning frameworks.

Repo: [[multilayer perceptron]](https://github.com/swangarch/multilayer_perceptron)

The agent learns to maximize its snake length through reinforcement learning, improving its policy via experience replay and Q-learning updates.

## Features

- Custom-built 10×10 Snake environment
- Implementation of a **Deep Q-Network (DQN)**  
- Neural network (MLP) built from scratch only using **NumPy**  
- Replay memory and epsilon-greedy exploration  
- Visualization of training progress and gameplay
- Differents training strategies for comparasion

## Dependencies

- Python  
- NumPy  
- Matplotlib
- Pygame

## Getting Started

### Setup

```bash
git clone https://github.com/swangarch/learn2slither.git
cd learn2slither

make setup
source venv/bin/activate # to setup virtual enviroment
```
### Train and play

```bash
make train-show 
# train with visualization

make train-load 
# train with visualization and pre-trained weights

make train-off 
# train without visualization

make train-off-load 
# train without visualization but with pre-trained weights

make play 
# evaluate the trained model (no random moves, loads pre-trained weights)

```
During training with visualization, press Q to slow down the progress for observation, and E to resume.

## Enviroment

The Snake game is played on a 10×10 grid, starting with 3 body segments, and two apples are randomly placed on the grid. When one apple is eaten, a new one is generated.

The neural network takes the current game state as input and predicts the Q-values for all four possible directions.

Two different environment representations are used in the DQN to compare performance:

- #### Input: Vector of coordinates
    Includes snake head, body parts, and food positions. Compact and efficient for small maps.

    Example:

    ```[c1.x1, c1.x2, c1.x1, c1.x2, s1.x1, s1.x2, s2.x1, s2.x2, s3.x1, s3.x2 ... ]```

    (28 dimensions total)
    
    Here, c1 and c2 represent the collectibles (apples), s1 is the head, and the remaining s values represent the body.
    If the snake length is less than 12, -1 padding is used.

    After training on 20,000 game sessions, the maximum achieved length is 22, with an average around 15–18.


- #### Input: Row and column encoding
    Example:

    ```[0, 0, C, 0, 0, 0, H, 0, 0, 0] + [0, 0, C, 0, H, S, S, 0, 0, 0]```

    (20 dimensions total)

    After training on 5,000 game sessions, the maximum achieved length is 33, with an average around 25–30.

    After training on 50,000 game sessions, the maximum achieved length is 45, with an average around 35–40.

Because the row and column encoding shows significantly better performance, it is used as the default representation in this project.

- ## Learning Strategies
    - Use a replay memory pool of size 5000 to store states and Q-value targets.
Each step samples a batch of 64 experiences for training.
    - Start with random actions, gradually decreasing epsilon to let model predictions take over, while retaining a small amount of exploration.
    - Since eating an apple is a low-probability event, successful experiences are duplicated and rewards are increased to balance the dataset, and eat apple will have bigger reward value.
