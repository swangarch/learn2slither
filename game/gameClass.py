import pygame
import numpy as np
from game20 import *
from .dqn import create_dqn
import random as rd
from collections import deque

class SnakeGame():
    def __init__(self):
        self.SIZE = 10
        self.render = True
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
        self.memlen = 1000
        self.states = deque(maxlen=self.memlen)
        self.actions = deque(maxlen=self.memlen)
        self.rewards = deque(maxlen=self.memlen)
        self.after_states = deque(maxlen=self.memlen)
        
        self.epsilon = 1


    def loop(self, running):
        # create a neural network
        dqn, conf = create_dqn()
        iteration = 0

        while running:
            # reset game logic
            site_state = self.reset_loop(iteration)
            if self.render == True:
                running, dirIdx = self.event_handler()
            if self.epsilon > 0.1:
                self.epsilon -= 0.0005
            # use neural network to choose the direction, based on site state
            if rd.random() > self.epsilon:
                currDirIdx = self.pred_direction(dqn, site_state)
            else:
                currDirIdx = rd.randint(0, 3)
            currDir = self.dirs[currDirIdx]

            # update head position based on the direction choosen by neural network
            new_pos = [self.player_pos[0] + currDir[0], self.player_pos[1] + currDir[1]]
            
            # update snake head position
            self.player_pos = new_pos
            # if snake head has collision with snake body, add punishment, and stop this turn
            self.check_self_collision(new_pos)
            #update snake node
            self.snake.insert(0, new_pos)
            # if snake hit a collectible, add new node to body
            self.collect_item(new_pos)
            # if snake out of boundary, add punishment
            self.check_out_of_bounds(new_pos)
            
            # if at the end of this turn, snake lost, init a new turn
            if self.reward == -0.9:
                self.snake, self.collectible, self.currDir, self.player_pos = init_state()
                iteration += 1
            
            # -------------------------------------------------
            # record history
            self.states.append(site_state)
            self.actions.append(currDirIdx)
            self.rewards.append(self.reward)
            self.after_states.append(create_matrix(self.snake, self.collectible, self.SIZE))

            # 转换成numpy数组
            states_array = np.array(self.states).squeeze(axis=-1)  # shape: [N, SIZE]
            actions_array = np.array(self.actions).reshape(-1, 1)  # shape: [N,]
            rewards_array = np.array(self.rewards).reshape(-1, 1)  # shape: [N,]
            next_states_array = np.array(self.after_states).squeeze(axis=-1)   # shape: [N, SIZE]

            # print(states_array.shape, actions_array.shape, rewards_array.shape, next_states_array.shape)

            Q_current = dqn.inference(states_array)  # shape: [N, 4]
            Q_next = dqn.inference(next_states_array)  # shape: [N, 4]

            # 计算目标Q值
            gamma = 0.95
            Q_target = Q_current.copy()  # 先复制一份

            for i in range(len(actions_array)):
                action = actions_array[i]
                reward = rewards_array[i]
                if reward == -0.9:
                    Q_target[i, action[0]] = reward
                else:
                    Q_target[i, action[0]] = reward + gamma * np.max(Q_next[i])
            

            for i in range(5):
                dqn.train_batch_rl(states_array, Q_target, 0.002)

            self.update_display(currDirIdx, iteration)
        if self.render == True:
            pygame.quit()
        dqn.save_plots()


    def reset_loop(self, iteration):
        if self.render == True:
            self.screen.fill("yellow")
        self.reward = -0.01
        # Update matrix, and object on matrix based on snake, and collectible state
        site_state = create_matrix(self.snake, self.collectible, self.SIZE)
        return site_state


    def check_self_collision(self, new_pos):
        for node in self.snake:
            if new_pos[0] == node[0] and new_pos[1] == node[1]:
                self.reward = -0.9
                break


    def collect_item(self, new_pos):
        hit_collectible = False
        for i, item in enumerate(self.collectible):
            if new_pos[0] == item[0] and new_pos[1] == item[1]:
                hit_collectible = True
                self.collectible.pop(i)
                self.reward = 2
                add_collectible(self.collectible, self.snake, self.SIZE)
                break
        # if not hit a collectible, remove the last node, to keep snake length
        if hit_collectible == False:
            self.snake.pop()

        
    def update_display(self, currDir, iteration):
        if self.render == True:
            draw_snake(self.screen, self.snake, self.radius)
            draw_item(self.screen, self.collectible, self.radius)
        print("[ITER]", iteration, "[DIR]", currDir, "[SCORE]", self.reward, "[MEM_LEN]", len(self.states), end="")
        print(" [SNAKE]", end="")
        for i in range(len(self.snake)):
            print("=", end="")
        print()
        if self.render == True:
            pygame.display.flip()
            self.clock.tick(self.tick_time)


    def pred_direction(self, dqn, site_state):
        predict_dir_idx = dqn.inference(site_state.T).argmax(axis=1, keepdims=True)[0][0]
        return predict_dir_idx

    
    def check_out_of_bounds(self, new_pos) -> bool:
        if new_pos[0] >= self.SIZE or new_pos[0] <= 0 or new_pos[1] >= self.SIZE or new_pos[1] <= 0:
            self.reward = -0.9
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
            elif event.key == pygame.K_q:
                if self.tick_time > 50:
                    self.tick_time -= 50
                else:
                    self.tick_time = 10
            elif event.key == pygame.K_e:
                self.tick_time += 50
            return dirIdx
        return None