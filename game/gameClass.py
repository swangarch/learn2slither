import pygame
import numpy as np
from game import *
from .dqn import DQN
import random as rd
import sys


class SnakeGame():
    def __init__(self, render=False, weights=None, randMove=True, trainMode=True):
        
        # game visual setting
        self.SIZE = 10
        self.render = render
        self.radius = 20
        self.tick_time = 5000
        if self.render == True:
            pygame.init()
            self.screen = pygame.display.set_mode((self.radius * 2 * self.SIZE + self.radius, self.radius * 2 * self.SIZE + self.radius))
            self.clock = pygame.time.Clock()

        # game states
        self.dirs  =  [(0, -1), (0, 1), (-1, 0),  (1, 0)]
        self.snake, self.collectible, self.currDir, self.player_pos = init_state()
        self.dirIdx = 0
        self.reward = 0
        self.collection_dist = None
        self.done = False
        # max length in all sessions
        self.max_len = 3

        # memory pool
        self.memlen_max = 5000
        self.states, self.actions, self.rewards, self.after_states, self.dones = init_mem_pool(self.memlen_max)
        
        # hyper parameters
        self.epsilon = 1
        self.decay = 0.9999
        self.gamma = 0.95
        self.batch_size = 64
        self.learning_rate = 0.002
        self.eat_reward = 5

        # train mode
        self.randMove = randMove
        self.trainMode = trainMode

        # neural network
        self.model = DQN.create_dqn()
        if weights is not None:
            self.model.load_weights(weights)


    def select_move_dir(self, site_state):
        self.epsilon = max(0.05, self.epsilon * self.decay)
        if self.randMove == False or rd.random() > self.epsilon:
            currDirIdx = self.pred_direction(self.model, site_state)
        else:
            currDirIdx = rd.randint(0, 3)
        currDir = self.dirs[currDirIdx]
        return currDir, currDirIdx


    def cal_Q_target(self, states_array, next_states_array, actions_array, rewards_array, is_dones):
        Q_current = self.model.inference(states_array)  # shape: [N, 4]
        Q_next = self.model.inference(next_states_array)  # shape: [N, 4]
        Q_target = Q_current
        for i in range(len(actions_array)):
            action = actions_array[i]
            reward = rewards_array[i]
            if is_dones[i] == True:
                Q_target[i, action[0]] = reward
            else:
                Q_target[i, action[0]] = reward + self.gamma * np.max(Q_next[i])
        return Q_target


    def add_to_mem_pool(self, site_state, currDirIdx):
        self.states.append(site_state)
        self.actions.append(currDirIdx)
        self.rewards.append(self.reward)
        after_state = self.build_state()
        self.after_states.append(after_state)
        self.dones.append(self.done)


    def adjust_lr(self, iteration):
        # if iteration < 5000:
        #     lr = self.learning_rate
        # elif iteration < 10000:
        #     lr = self.learning_rate /  2.0
        # else:
        #     lr = self.learning_rate /  4.0
        return self.learning_rate


    def build_state(self):
        states = np.zeros(20, dtype=np.float32)
        for c in self.collectible:
            if c[0] == self.snake[0][0]:
                states[c[1]] = -1.0
            if c[1] == self.snake[0][1]:
                states[c[0] + 10] = -1.0
        for node in self.snake[1:]:
            if node[0] == self.snake[0][0]:
                states[node[1]] = 0.5
            if node[1] == self.snake[0][1]:
                states[node[0] + 10] = 0.5
        states[self.snake[0][1]] = 1
        states[self.snake[0][0] + 10] = 1
        return states.reshape(-1, 1)


    def dup_key_mem(self, vec_state, currDirIdx):
        if self.reward > 5:
            self.add_to_mem_pool(vec_state, currDirIdx)
            self.add_to_mem_pool(vec_state, currDirIdx)


    def training(self, move_count, iteration):
         if len(self.states) > 1000:
            indices = rd.sample(range(len(self.states)), min(self.batch_size, len(self.states)))
            states_array = np.array([self.states[i] for i in indices]).squeeze(-1)
            next_states_array = np.array([self.after_states[i] for i in indices]).squeeze(-1)
            actions_array = np.array([self.actions[i] for i in indices]).reshape(-1, 1)
            rewards_array = np.array([self.rewards[i] for i in indices]).reshape(-1, 1)
            dones_array = np.array([self.dones[i] for i in indices])

            Q_target = self.cal_Q_target(states_array, next_states_array, actions_array, rewards_array, dones_array)
            self.model.train_batch_rl(move_count, states_array, Q_target, self.adjust_lr(iteration))


    def move(self, currDir):
        new_pos = [self.player_pos[0] + currDir[0], self.player_pos[1] + currDir[1]]
        return new_pos


    def loop(self, running):
        iteration = 0
        move_count = 0
        try:
            while running:
                self.reset_loop()
                if self.render == True:
                    running, dirIdx = self.event_handler()
                vec_state = self.build_state()
                currDir, currDirIdx = self.select_move_dir(vec_state)
                iteration += self.handle_step(self.move(currDir))

                if self.trainMode == True:
                    self.dup_key_mem(vec_state, currDirIdx)
                    self.add_to_mem_pool(vec_state, currDirIdx)
                    self.training(move_count, iteration)
                self.update_display()
                iteration = self.train_log(currDirIdx, iteration)
                move_count += 1
        except KeyboardInterrupt as e:
            pass
        if self.render == True:
            pygame.quit()
        self.model.close_visual()
        self.model.save_plots()
        if self.trainMode == True:
            self.model.save_weights()


    def reset_loop(self):
        if self.render == True:
            self.screen.fill("yellow")
        self.reward = 0
        if self.done == True:
            self.collection_dist = None
        self.done = False


    def check_self_collision(self, new_pos):
        if new_pos in self.snake:
            self.reward = -1
            self.done = True


    def handle_step(self, new_pos):
        #Check if game end
        self.check_self_collision(new_pos)
        self.check_out_of_bounds(new_pos)
        # if game end init, otherwise try to collect item, add reward if snake gets closer to apple
        if self.done != True:
            self.collect_item(new_pos)
            if self.reward < self.eat_reward / 2.0:
                self.reward -= 0.01
            curr_collection_dist = min(dist(self.collectible[0], new_pos), dist(self.collectible[1], new_pos))
            if self.collection_dist is None:
                self.collection_dist = curr_collection_dist
            elif self.collection_dist > curr_collection_dist:
                if self.reward < self.eat_reward / 2.0:
                    self.reward += min(0.02, 0.002 * (self.collection_dist - curr_collection_dist))
                self.collection_dist = curr_collection_dist
            else:
                if self.reward < self.eat_reward / 2.0:
                    self.reward += max(-0.02, 0.002 * (self.collection_dist - curr_collection_dist))
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
                self.reward = self.eat_reward
                add_collectible(self.collectible, self.snake, self.SIZE)
                break
        # if not hit a collectible, remove the last node, to keep snake length
        if hit_collectible == False:
            self.snake.pop()
        self.player_pos = new_pos


    def train_log(self, currDir, iteration):
        len_snake = len(self.snake)
        if len_snake > self.max_len:
            self.max_len = len_snake
        snake = ''.join(('<' if i == 0 else '-') for i in range(len_snake))
        print(f"[ITER] {iteration:4d} [DIR] {currDir} [REWARD] {self.reward:6.2f} [MEM_LEN] {len(self.states):4d} [SNAKE] {self.max_len}  {snake}")
        return iteration


    def update_display(self):
        if self.render == True:
            draw_snake(self.screen, self.snake, self.radius)
            draw_item(self.screen, self.collectible, self.radius)
            pygame.display.flip()
            self.clock.tick(self.tick_time)


    def pred_direction(self, dqn, site_state):
        Q_curr = dqn.inference(site_state.T)
        predict_dir_idx = Q_curr.argmax(axis=1, keepdims=True)[0][0]
        return predict_dir_idx


    def check_out_of_bounds(self, new_pos) -> bool:
        if new_pos[0] >= self.SIZE or new_pos[0] < 0 or new_pos[1] >= self.SIZE or new_pos[1] < 0:
            self.reward = -1
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
            if event.key == pygame.K_w: 
                dirIdx = 0 # dirUp
            elif event.key == pygame.K_s:
                dirIdx = 1 # dirDown
            elif event.key == pygame.K_a:
                dirIdx = 2 # dirLeft
            elif event.key == pygame.K_d:
                dirIdx = 3 # dirRight
            elif event.key == pygame.K_q:
                self.tick_time = 5
            elif event.key == pygame.K_e:
                self.tick_time = 5000
            return dirIdx
        return None