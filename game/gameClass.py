import pygame
import numpy as np
from numpy import ndarray as array
from game import *
from .dqn import DQN
import random as rd
import sys


class SnakeGame():
    def __init__(self, render=False, weights=None, rand_move=True, train_mode=True):
        
        # game visual setting
        self.SIZE = 10
        self.render = render
        self.radius = 30
        self.tick_time = 5000
        if self.render == True:
            pygame.init()
            x_size = self.radius * 2 * self.SIZE + self.radius
            y_size = self.radius * 2 * self.SIZE + self.radius
            self.screen = pygame.display.set_mode((x_size + 300, y_size))
            pygame.display.set_caption("Duck 2 slither")
            self.clock = pygame.time.Clock()
            self.load_assets(x_size, y_size)

        # game states
        self.dirs  =  [(0, -1), (0, 1), (-1, 0),  (1, 0)]
        self.snake, self.collectible, self.bad_collectible, self.currDir, self.player_pos = init_state()
        self.dirIdx = 0
        self.reward = 0
        self.done = False
        self.state = np.zeros(20, dtype=np.float32)

        # game dashboard
        self.lifetime = 0
        self.final_score = 0
        self.max_len = 3 # max length in all sessions
        self.max_final_score = 0

        # memory pool
        self.memlen_max = 5000
        self.states, self.actions, self.rewards, self.after_states, self.dones = init_mem_pool(self.memlen_max)
        
        # hyper parameters
        self.decay = 0.9999
        self.gamma = 0.95
        self.batch_size = 64
        self.learning_rate = 0.002
        if rand_move == False:
            self.epsilon = 0.005
            self.min_exploration_rate = 0.005
        else:
            self.epsilon = 1
            self.min_exploration_rate = 0.05

        # reward and penality setting
        self.eat_reward = 10
        self.death_penalty = -1
        self.lazy_penality = -0.01
        self.eat_penality = -0.2

        # train mode
        self.rand_move = rand_move
        self.train_mode = train_mode

        # neural network
        self.model = DQN.create_dqn()
        if weights is not None:
            self.model.load_weights(weights)
    

    def load_assets(self, x_size, y_size):
        self.font1 = pygame.font.Font("./game/assets/fonts/Bungee-Regular.ttf", 13)
        self.font2 = pygame.font.Font("./game/assets/fonts/Bungee-Regular.ttf", 18)
        self.font3 = pygame.font.Font("./game/assets/fonts/Chewy-Regular.ttf", 36)
        img = pygame.image.load("./game/assets/images/background.png")
        self.background_img = pygame.transform.scale(img, (x_size, y_size))
        img_red = pygame.image.load("./game/assets/images/fox.png")
        self.red_img = pygame.transform.scale(img_red, (2 * self.radius, 2 * self.radius))
        img_green = pygame.image.load("./game/assets/images/egg.png")
        self.green_img = pygame.transform.scale(img_green, (2 * self.radius, 2 * self.radius))
        img_head = pygame.image.load("./game/assets/images/duck1.png")
        self.head_img1 = pygame.transform.scale(img_head, (2 * self.radius, 2 * self.radius))
        img_head = pygame.image.load("./game/assets/images/duck2.png")
        self.head_img2 = pygame.transform.scale(img_head, (2 * self.radius, 2 * self.radius))
        img_body = pygame.image.load("./game/assets/images/duck_baby.png")
        self.body_img = pygame.transform.scale(img_body, (2 * self.radius, 2 * self.radius))


    def run(self, max_iter=10000):
        iteration = 0
        move_count = 0
        running = True
        try:
            while running and iteration < max_iter:
                self.reset_loop()
                if self.render == True:
                    running, dirIdx = self.event_handler()
                vec_state = self.build_state()
                curr_dir, curr_dirIdx = self.select_move_dir(vec_state)
                iteration += self.handle_step(self.move(curr_dir))

                if self.train_mode == True:
                    self.dup_key_mem(vec_state, curr_dirIdx)
                    self.add_to_mem(vec_state, curr_dirIdx)
                    self.train(move_count)
                self.update_display(curr_dirIdx, iteration)
                iteration = self.train_log(curr_dirIdx, iteration)
                move_count += 1
                self.lifetime += 1
                self.final_score += self.reward
        except KeyboardInterrupt as e:
            pass
        if self.render == True:
            pygame.quit()
        self.model.save_plot()
        self.model.close_visual()
        if self.train_mode == True:
            self.model.save_weights()


    def reset_loop(self):
        if self.render == True:
            self.screen.fill((138, 190, 185))
        self.reward = 0
        if self.done == True:
            self.lifetime = 0
            if self.max_final_score < self.final_score:
                self.max_final_score = self.final_score
            self.final_score = 0
        self.done = False


    def train(self, move_count):
        if len(self.states) > 1000:
            indices = rd.sample(range(len(self.states)), min(self.batch_size, len(self.states)))
            states_array = np.array([self.states[i] for i in indices]).squeeze(-1)
            next_states_array = np.array([self.after_states[i] for i in indices]).squeeze(-1)
            actions_array = np.array([self.actions[i] for i in indices]).reshape(-1, 1)
            rewards_array = np.array([self.rewards[i] for i in indices]).reshape(-1, 1)
            dones_array = np.array([self.dones[i] for i in indices])

            Q_target = self.cal_Q_target(states_array, next_states_array, actions_array, rewards_array, dones_array)
            self.model.train_batch_rl(move_count, states_array, Q_target, self.learning_rate)


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


    def vec_repr(self, collection: list, value: float):
        for c in collection:
            if c[0] == self.snake[0][0]:
                self.state[c[1]] = value
            if c[1] == self.snake[0][1]:
                self.state[c[0] + 10] = value


    def build_state(self):
        self.state *= 0
        if len(self.snake) == 0:
            return self.state.reshape(-1, 1)
        self.vec_repr(self.collectible, -0.4)
        self.vec_repr(self.bad_collectible, -1.0)
        self.vec_repr(self.snake[1:], 0.6)
        self.state[self.snake[0][1]] = 1.0
        self.state[self.snake[0][0] + 10] = 1.0
        return self.state.reshape(-1, 1).copy()
    

    def dup_key_mem(self, vec_state, currDirIdx):
        if self.reward > self.eat_reward / 2.0:
            self.add_to_mem(vec_state, currDirIdx)
            self.add_to_mem(vec_state, currDirIdx)


    def add_to_mem(self, site_state, currDirIdx):
        self.states.append(site_state)
        self.actions.append(currDirIdx)
        self.rewards.append(self.reward)
        after_state = self.build_state()
        self.after_states.append(after_state)
        self.dones.append(self.done)


    def pred_direction(self, dqn, site_state):
        Q_curr = dqn.inference(site_state.T)
        predict_dir_idx = Q_curr.argmax(axis=1, keepdims=True)[0][0]
        return predict_dir_idx


    def select_move_dir(self, site_state):
        self.epsilon = max(self.min_exploration_rate, self.epsilon * self.decay)
        if rd.random() > self.epsilon:
            currDirIdx = self.pred_direction(self.model, site_state)
        else:
            currDirIdx = rd.randint(0, 3)
        currDir = self.dirs[currDirIdx]
        return currDir, currDirIdx


    def move(self, currDir):
        new_pos = [
                    self.player_pos[0] + currDir[0], 
                    self.player_pos[1] + currDir[1]
                ]
        return new_pos


    def check_self_collision(self, new_pos):
        if new_pos in self.snake:
            self.reward = self.death_penalty 
            self.done = True


    def check_out_of_bounds(self, new_pos) -> bool:
        if new_pos[0] >= self.SIZE or new_pos[0] < 0 or new_pos[1] >= self.SIZE or new_pos[1] < 0:
            self.reward = self.death_penalty 
            self.done = True
            return True
        return False


    def handle_step(self, new_pos):
        # Check if game end
        self.check_self_collision(new_pos)
        self.check_out_of_bounds(new_pos)
        # if game end init, otherwise try to collect item, add reward if snake gets closer to apple
        if self.done != True:
            done = self.collect_item(new_pos)
            if done == True:
                return 0
            if self.reward == 0:
                self.reward += self.lazy_penality
            return 0
        else:
            self.snake, self.collectible, self.bad_collectible, self.currDir, self.player_pos = init_state()
            return 1


    def collect_item(self, new_pos) -> bool: # return if snake dead
        self.snake.insert(0, new_pos)
        hit_collectible = False
        for i, item in enumerate(self.collectible):
            if new_pos[0] == item[0] and new_pos[1] == item[1]:
                hit_collectible = True
                self.collectible.pop(i)
                self.reward = self.eat_reward
                add_collectible(self.collectible, self.snake, self.bad_collectible, self.SIZE)
                break
        hit_bad_collectible = False
        for i, item in enumerate(self.bad_collectible):
            if new_pos[0] == item[0] and new_pos[1] == item[1]:
                hit_bad_collectible = True
                self.bad_collectible.pop(i)
                self.reward = self.eat_penality
                add_collectible(self.bad_collectible, self.snake, self.collectible, self.SIZE)
                break
        # if not hit a collectible, remove the last node, to keep snake length
        if hit_collectible == False:
            self.snake.pop()
        if hit_bad_collectible == True:
            if len(self.snake) == 0:
                self.reward = self.death_penalty
                self.done = True
                return True
            else:
                self.snake.pop()
        self.player_pos = new_pos
        return False


    def render_text_block(self, pos, title:str, content:list):
        color1 = (48, 86, 105)
        color2 = (193, 120, 90)
        interval = 25
        rel_pos = interval
        draw_text(self.screen, self.font2, title, (pos[0] + 1, pos[1] + 1), color1)
        draw_text(self.screen, self.font2, title, (pos[0], pos[1]), color2)
        for c in content:
            draw_text(self.screen, self.font1, c, (pos[0], pos[1] + rel_pos), color2)
            rel_pos += interval


    def render_dir(self, currDirIdx):
        dir_posX, dir_posY = 700, 550
        size = 50
        interval = size + 2
        color_bg = (163, 189, 165)
        color_active = (193, 120, 90)
        color_shadow = (48, 86, 105)
        colors = [color_bg, color_bg, color_bg, color_bg]
        colors[currDirIdx] = color_active

        pygame.draw.rect(self.screen, color_shadow, (dir_posX + interval + 2, dir_posY - interval + 2, size, size), border_radius=15)
        pygame.draw.rect(self.screen, color_shadow, (dir_posX + 2, dir_posY + 2, size, size), border_radius=15)
        pygame.draw.rect(self.screen, color_shadow, (dir_posX + interval + 2, dir_posY + 2, size, size), border_radius=15)
        pygame.draw.rect(self.screen, color_shadow, (dir_posX + interval * 2 + 2, dir_posY + 2, size, size), border_radius=15)

        # -----------------------------------------------------------------------------------
        pygame.draw.rect(self.screen, colors[0], 
                            (dir_posX + interval, dir_posY - interval, size, size), 
                             border_radius=15
                        )
        pygame.draw.rect(self.screen, colors[2], 
                             (dir_posX, dir_posY, size, size), 
                             border_radius=15
                        )
        pygame.draw.rect(self.screen, colors[1], 
                            (dir_posX + interval, dir_posY, size, size),
                            border_radius=15
                        )
        pygame.draw.rect(self.screen, colors[3], 
                            (dir_posX + interval * 2, dir_posY, size, size),
                            border_radius=15
                        )


    def render_bar(self, posX, posY, color, value, max):
        bar_size = 180
        bar_len = value / max * bar_size
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        (posX , posY, bar_size, 3)
                    )
        pygame.draw.rect(self.screen, color, 
                        (posX , posY, bar_len, 3)
                    )


    def render_view(self, posX, posY):
        color_map = {
            -4: (141, 170, 136),
            -10: (187, 110, 107),
            6: (120, 155, 180),
            10: (202, 141, 91), 
            0: (255, 255, 255)
        }
        bar_len = 18
        for i, item in enumerate(self.state[:10]):
            pygame.draw.rect(self.screen, color_map[round(float(item) * 10)], (posX + i * bar_len, posY, bar_len, 3))
        for i, item in enumerate(self.state[10:]):
            pygame.draw.rect(self.screen, color_map[round(float(item) * 10)], (posX + i * bar_len, posY + 10, bar_len, 3))


    def update_display(self, currDirIdx, iteration):
        if self.render == True:
            self.screen.blit(self.background_img, (0, 0))

            duck_img = self.head_img1 if currDirIdx % 2 == 1 else self.head_img2
            draw_snake(self.screen, self.snake, self.radius, duck_img, self.body_img)
            draw_item(self.screen, self.collectible, self.bad_collectible, self.radius, self.green_img, self.red_img)
            
            posX, posY = 680, 30
            color_bg = (183, 229, 205)
            color1 = (48, 86, 105)
            color2 = (193, 120, 90)

            pygame.draw.rect(self.screen, color1, (posX - 30 + 5, posY - 20 + 5, 260, 610), width=0, border_radius=15)
            pygame.draw.rect(self.screen, color_bg, (posX - 30, posY - 20, 260, 610), width=0, border_radius=15)
            
            draw_text(self.screen, self.font3, "Duck 2 slither", (posX + 2, posY + 2), color1)
            draw_text(self.screen, self.font3, "Duck 2 slither", (posX, posY), color2)
            self.render_text_block((posX, posY + 70), "Mode", [
                f"{'Training' if self.train_mode else 'Playing'}"
            ])
            self.render_text_block((posX, posY + 130), "Stage", [
                f"Lifetime  {self.lifetime}", f"Snake Size  {len(self.snake)}",
                f"Instant Reward  {self.reward}", f"Final Score  {self.final_score:.2f}",
                f"View",
            ])

            explo_rate = max(self.epsilon, self.min_exploration_rate)
            self.render_text_block((posX, posY + 300), "Record", [
                f"Game Session  {iteration}", f"Max Length  {self.max_len}",
                f"Max Score  {self.max_final_score:.2f}",
                f"Memory Pool  {len(self.states)} / 5000",
                f"Exploration Rate  {explo_rate:.2f}",
            ])
            self.render_view(posX, posY + 280)
            self.render_bar(posX, posY + 370, color2, self.max_len, 60)
            self.render_bar(posX, posY + 420, color2, len(self.states), 5000)
            self.render_bar(posX, posY + 445, color2, explo_rate, 1)
            self.render_dir(currDirIdx)
            pygame.display.flip()
            self.clock.tick(self.tick_time)


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


    def train_log(self, currDir, iteration):
        len_snake = len(self.snake)
        if len_snake > self.max_len:
            self.max_len = len_snake
        snake = ''.join(('<' if i == 0 else '-') for i in range(len_snake))
        print(f"[ITER] {iteration:4d} [DIR] {currDir} [REWARD] {self.reward:6.2f} [MEM_LEN] {len(self.states):4d} [SNAKE] {self.max_len}  {snake}")
        return iteration