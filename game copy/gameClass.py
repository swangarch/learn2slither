import pygame
import numpy as np
from game import *
from .dqn import create_dqn
import random as rd

class SnakeGame():
    def __init__(self):
        self.SIZE = 10
        pygame.init()
        self.dirs  =  [(0, -1), (0, 1), (-1, 0),  (1, 0)]
        self.ground_size = 20

        self.snake, self.collectible, self.currDir, self.player_pos = init_state()
        self.dirIdx = 0
        self.radius = 10
        self.screen = pygame.display.set_mode((self.ground_size * self.SIZE, self.ground_size * self.SIZE))
        self.clock = pygame.time.Clock()
        self.running = True
        self.hit = False
        self.tick_time = 10
        self.score = 0
    
        self.site_state = create_matrix(self.snake, self.collectible, self.SIZE)
        print(self.site_state.reshape((self.SIZE, self.SIZE)))

        # memory pool
        self.states = []
        self.actions = []
        self.scores = []
        self.after_states = []


    def loop(self, running):

        # create a neural network
        dqn, conf = create_dqn()
        iteration = 0

        while running:
            # reset game logic

            site_state = self.reset_loop(iteration)
            running, dirIdx = self.event_handler()

            # use neural network to choose the direction, based on site state
            if iteration % 5 == 1:
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
            if self.score == -0.9:
                self.snake, self.collectible, self.currDir, self.player_pos = init_state()
                self.score += -0.9
                iteration += 1

            # record history
            self.states.append(site_state) #!!!!!!!!!!!!!!!!!!!!!
            self.actions.append(currDirIdx) #!!!!!!!!!!!!!!!!!!!!
            self.scores.append(self.score) #!!!!!!!!!!!!!!!!!!!!!
            self.after_states.append(create_matrix(self.snake, self.collectible, self.SIZE)) #!!!!!!!!!!!!!!!!!!!!!!!!!

            # train the model using history
            if iteration % 5 == 1:
                for i in range(100):
                    dqn.train_batch_rl(np.squeeze(np.array(self.states), axis=-1), 
                                    np.array(self.scores).reshape(-1, 1), 
                                    0.00001)

            # update display
            self.update_display()
        pygame.quit()
        dqn.save_plots()


    def reset_loop(self, iteration):
        if iteration % 2 == 0:
            self.screen.fill("yellow")
        else:
            self.screen.fill("cyan")
        self.score = -0.1
        # Update matrix, and object on matrix based on snake, and collectible state
        site_state = create_matrix(self.snake, self.collectible, self.SIZE)
        return site_state


    def check_self_collision(self, new_pos):
        for node in self.snake:
            if new_pos[0] == node[0] and new_pos[1] == node[1]:
                self.score = -0.9
                break


    def collect_item(self, new_pos):
        hit_collectible = False
        for i, item in enumerate(self.collectible):
            if new_pos[0] == item[0] and new_pos[1] == item[1]:
                hit_collectible = True
                self.collectible.pop(i)
                self.score = 1
                add_collectible(self.collectible, self.snake, self.SIZE)
                break
        # if not hit a collectible, remove the last node, to keep snake length
        if hit_collectible == False:
            self.snake.pop()

        
    def update_display(self):
        draw_snake(self.screen, self.snake, self.radius)
        draw_item(self.screen, self.collectible, self.radius)
        print("[SCORE]", self.score, "[MEM_LEN]", len(self.states))
        pygame.display.flip()
        self.clock.tick(self.tick_time)


    def pred_direction(self, dqn, site_state):
        predict_dir_idx = dqn.inference(site_state.T).argmax(axis=1, keepdims=True)[0][0]
        print(predict_dir_idx)
        return predict_dir_idx

    
    def check_out_of_bounds(self, new_pos) -> bool:
        if new_pos[0] >= self.SIZE or new_pos[0] <= 0 or new_pos[1] >= self.SIZE or new_pos[1] <= 0:
            self.score = -0.9
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
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w: #and player_pos[1] - 1 >= 0: 
                dirIdx = 0 # dirUp
            elif event.key == pygame.K_s: #and player_pos[1] + 1 <= ground_size:
                dirIdx = 1 # dirDown
            elif event.key == pygame.K_a: # and player_pos[0] - 1 >= 0:
                dirIdx = 2 # dirLeft
            elif event.key == pygame.K_d: # and player_pos[0] + 1 <= ground_size:
                dirIdx = 3 # dirRight
            return dirIdx
        return None