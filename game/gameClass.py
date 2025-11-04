import pygame
import numpy as np
from game import *
from .dqn import create_dqn
import random as rd
from collections import deque
import sys


class SnakeGame():
    def __init__(self, render=False, weights=None):
        self.SIZE = 10
        self.render = render
        self.dirs  =  [(0, -1), (0, 1), (-1, 0),  (1, 0)]
        self.ground_size = 20
        self.snake, self.collectible, self.currDir, self.player_pos = init_state()
        self.dirIdx = 0
        self.radius = 10
        if self.render == True:
            pygame.init()
            self.screen = pygame.display.set_mode((self.ground_size * self.SIZE, self.ground_size * self.SIZE))
            self.clock = pygame.time.Clock()
        self.running = True
        self.hit = False
        self.tick_time = 5000
        self.reward = 0
        self.site_state = create_matrix(self.snake, self.collectible, self.SIZE)
        print(self.site_state.reshape((self.SIZE, self.SIZE)))

        # memory pool
        self.memlen_max = 500
        self.states = deque(maxlen=self.memlen_max)
        self.actions = deque(maxlen=self.memlen_max)
        self.rewards = deque(maxlen=self.memlen_max)
        self.after_states = deque(maxlen=self.memlen_max)
        self.dones = deque(maxlen=self.memlen_max)
        
        self.epsilon = 1
        self.gamma = 0.9

        self.collection_dist = None
        self.done = False
        self.model, conf = create_dqn()
        if weights is not None:
            self.model.load_weights(weights)
        self.max_len = 3


    def select_move_dir(self, site_state):
        if rd.random() > self.epsilon:
            currDirIdx = self.pred_direction(self.model, site_state)
        else:
            currDirIdx = rd.randint(0, 3)
        currDir = self.dirs[currDirIdx]
        return currDir, currDirIdx


    def cal_Q_target(self, states_array, next_states_array, actions_array, rewards_array):
        Q_current = self.model.inference(states_array)  # shape: [N, 4]
        Q_next = self.model.inference(next_states_array)  # shape: [N, 4]
        Q_target = Q_current.copy()
        for i in range(len(actions_array)):
            action = actions_array[i]
            reward = rewards_array[i]
            if self.dones[i] == True:
                Q_target[i, action[0]] = reward
            else:
                Q_target[i, action[0]] = reward + self.gamma * np.max(Q_next[i])
        return Q_target


    def add_to_mem_pool(self, site_state, currDirIdx):
        self.states.append(site_state)
        self.actions.append(currDirIdx)
        self.rewards.append(self.reward)
        self.after_states.append(create_matrix(self.snake, self.collectible, self.SIZE))
        self.dones.append(self.done)

        # convert to numpy arr
        states_array = np.array(self.states).squeeze(axis=-1)  # shape: [N, SIZE]
        actions_array = np.array(self.actions).reshape(-1, 1)  # shape: [N,]
        rewards_array = np.array(self.rewards).reshape(-1, 1)  # shape: [N,]
        next_states_array = np.array(self.after_states).squeeze(axis=-1)   # shape: [N, SIZE]
        return states_array, next_states_array, actions_array, rewards_array


    def get_training_batch(self, inputs, target, batch_size=100):
        total_samples = inputs.shape[0]
    
        # random choose some sample for mini batch training, or use all sample, if ther are not many
        if total_samples <= batch_size:
            sampled_inputs = inputs
            sampled_target = target
        else:
            indices = np.random.choice(total_samples, batch_size, replace=False)
            sampled_inputs = inputs[indices]
            sampled_target = target[indices]
        return sampled_inputs, sampled_target

    def adjust_lr(self, iteration):
        if iteration > 10000:
            lr = 0.001
        elif iteration > 5000:
            lr = 0.002
        else:
            lr = 0.005
        return lr

    def loop(self, running):
        iteration = 0
        while running:
            try:
                site_state = self.reset_loop(iteration)
                if self.render == True:
                    running, dirIdx = self.event_handler()

                # determine if choose random step or choose a step predicted by dqn
                self.epsilon = max(0.05, self.epsilon * 0.995)

                currDir, currDirIdx = self.select_move_dir(site_state)
                new_pos = [self.player_pos[0] + currDir[0], self.player_pos[1] + currDir[1]]
                iteration += self.handle_step(new_pos)

                states_array, next_states_array, actions_array, rewards_array = self.add_to_mem_pool(site_state, currDirIdx)
                Q_target = self.cal_Q_target(states_array, next_states_array, actions_array, rewards_array)
                if len(self.states) > 100:
                    sample_inputs, sample_targets = self.get_training_batch(states_array, Q_target)
                    self.model.train_batch(sample_inputs, sample_targets, self.adjust_lr(iteration))
                self.update_display(currDirIdx, iteration)
            except KeyboardInterrupt as e:
                break
    
        if self.render == True:
            pygame.quit()
        self.model.save_plots()
        self.model.save_weights()


    def reset_loop(self, iteration):
        if self.render == True:
            self.screen.fill("yellow")
        self.reward = 0
        if self.done == True:
            self.collection_dist = None
        self.done = False
        # Update matrix, and object on matrix based on snake, and collectible state
        site_state = create_matrix(self.snake, self.collectible, self.SIZE)
        
        return site_state


    def check_self_collision(self, new_pos):
        if new_pos in self.snake:
            self.reward = -50
            self.done = True

    @staticmethod
    def dist(loc1, loc2):
        return ((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)**0.5

    def handle_step(self, new_pos):
        #Check if game end
        self.check_self_collision(new_pos)
        self.check_out_of_bounds(new_pos)

        # if game end init, otherwise try to collect item, add reward if snake gets closer to apple
        if self.done != True:
            self.collect_item(new_pos)
            curr_collection_dist = min(SnakeGame.dist(self.collectible[0], new_pos), SnakeGame.dist(self.collectible[1], new_pos))
            if self.collection_dist is None:
                self.collection_dist = curr_collection_dist
            elif self.collection_dist > curr_collection_dist:
                self.reward += min(5, 0.1 * (self.collection_dist - curr_collection_dist))
                self.collection_dist = curr_collection_dist
            else:
                self.reward += min(5, 0.1 * (self.collection_dist - curr_collection_dist))
            return 0
        else:
            self.snake, self.collectible, self.currDir, self.player_pos = init_state()
            return 1


    def collect_item(self, new_pos):
        self.snake.insert(0, new_pos)
        hit_collectible = False
        for i, item in enumerate(self.collectible):
            if new_pos[0] == item[0] and new_pos[1] == item[1]:
                hit_collectible = True
                self.collectible.pop(i)
                self.reward = 20
                add_collectible(self.collectible, self.snake, self.SIZE)
                # self.collection_dist = None
                break
        # if not hit a collectible, remove the last node, to keep snake length
        if hit_collectible == False:
            self.snake.pop()
        self.player_pos = new_pos

        
    def update_display(self, currDir, iteration):
        if self.render == True:
            draw_snake(self.screen, self.snake, self.radius)
            draw_item(self.screen, self.collectible, self.radius)
        if self.done == True:
            iteration -= 1
        print(f"[ITER] {iteration:4d} [DIR] {currDir} [REWARD] {self.reward:6.2f} [MEM_LEN] {len(self.states):4d}", end="")
        print(" [SNAKE] ", end="")
        len_snake = len(self.snake)
        if len_snake > self.max_len:
            self.max_len = len_snake
        print(self.max_len, " ", end="")
        for i in range(len_snake):
            if i == 0:
                print("<", end="")
            else:   
                print("-", end="")
        print()
        if self.render == True:
            pygame.display.flip()
            self.clock.tick(self.tick_time)


    def pred_direction(self, dqn, site_state):
        predict_dir_idx = dqn.inference(site_state.T).argmax(axis=1, keepdims=True)[0][0]
        return predict_dir_idx

    
    def check_out_of_bounds(self, new_pos) -> bool:
        if new_pos[0] >= self.SIZE or new_pos[0] <= 0 or new_pos[1] >= self.SIZE or new_pos[1] <= 0:
            self.reward = -50
            self.done = True
            return True
        return False
    

    def event_handler(self):
        dirIdx = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, dirIdx
            # manual control logic is not used
            dirIdx = self.manual_control(event)
        return True, dirIdx


    def manual_control(self, event):
        dirIdx = -1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w: #and player_pos[1] - 1 >= 0: 
                dirIdx = 0 # dirUp
            elif event.key == pygame.K_s: #and player_pos[1] + 1 <= ground_size:
                dirIdx = 1 # dirDown
            elif event.key == pygame.K_a: # and player_pos[0] - 1 >= 0:
                dirIdx = 2 # dirLeft
            elif event.key == pygame.K_d: # and player_pos[0] + 1 <= ground_size:
                dirIdx = 3 # dirRight
            elif event.key == pygame.K_q: # slow down movement for observation
                self.tick_time = 5
            elif event.key == pygame.K_e:
                self.tick_time = 5000
            return dirIdx
        return None