import pygame
import numpy as np
from game import * 
from .IGame import IGame


class SnakeGame(IGame):

    color_map = {
            -4: (141, 170, 136),
            -10: (187, 110, 107),
            6: (120, 155, 180),
            10: (202, 141, 91), 
            0: (255, 255, 255)
        }
    
    color1 = (48, 86, 105)
    color2 = (193, 120, 90)
    color_bg = (183, 229, 205)
    color_button = (163, 189, 165)

    def __init__(self, render=True, train_mode=True):

        self._train_mode = train_mode
        # game visual setting
        self._SIZE = 10
        self._render = render
        self._radius = 30
        self._tick_time = 5000
        if self._render == True:
            pygame.init()
            x_size = self._radius * 2 * self._SIZE + self._radius
            y_size = self._radius * 2 * self._SIZE + self._radius
            self._screen = pygame.display.set_mode((x_size + 300, y_size))
            pygame.display.set_caption("Duck 2 slither")
            self._clock = pygame.time.Clock()
            self.load_assets(x_size, y_size)

        # game states
        self._dirs  =  [(0, -1), (0, 1), (-1, 0),  (1, 0)]
        self._snake, self._collectible, self._bad_collectible, self._currDir, self._player_pos = init_state()
        self._dirIdx = 0
        self._reward = 0
        self._done = False
        self._state = np.zeros(20, dtype=np.float32)

        # game dashboard
        self._lifetime = 0
        self._final_score = 0
        self._max_len = 3 # max length in all sessions
        self._session_max_len = 3
        self._max_final_score = 0
        self._total_len = 0
        
        self._eat_reward = 10
        self._death_penalty = -1
        self._lazy_penality = -0.01
        self._eat_penality = -0.2
    

    def load_assets(self, x_size, y_size):
        self._font1 = pygame.font.Font("./game/assets/fonts/Bungee-Regular.ttf", 13)
        self._font2 = pygame.font.Font("./game/assets/fonts/Bungee-Regular.ttf", 18)
        self._font3 = pygame.font.Font("./game/assets/fonts/Chewy-Regular.ttf", 36)
        img = pygame.image.load("./game/assets/images/background.png")
        self._background_img = pygame.transform.scale(img, (x_size, y_size))
        img_red = pygame.image.load("./game/assets/images/fox.png")
        self._red_img = pygame.transform.scale(img_red, (2 * self._radius, 2 * self._radius))
        img_green = pygame.image.load("./game/assets/images/egg.png")
        self._green_img = pygame.transform.scale(img_green, (2 * self._radius, 2 * self._radius))
        img_head = pygame.image.load("./game/assets/images/duck1.png")
        self._head_img1 = pygame.transform.scale(img_head, (2 * self._radius, 2 * self._radius))
        img_head = pygame.image.load("./game/assets/images/duck2.png")
        self._head_img2 = pygame.transform.scale(img_head, (2 * self._radius, 2 * self._radius))
        img_body = pygame.image.load("./game/assets/images/duck_baby.png")
        self._body_img = pygame.transform.scale(img_body, (2 * self._radius, 2 * self._radius))


    def quit_game(self):
        if self._render == True:
            pygame.quit()


    def reset(self):
        running = True
        if self._render == True:
            self._screen.fill((138, 190, 185))
            running, dirIdx = self.event_handler()
        self._reward = 0
        if self._done == True:
            self._total_len += self._session_max_len
            self._session_max_len = 3
            self._lifetime = 0
            if self._max_final_score < self._final_score:
                self._max_final_score = self._final_score
            self._final_score = 0
        self._done = False
        return running


    def vec_repr(self, collection: list, value: float):
        for c in collection:
            if c[0] == self._snake[0][0]:
                self._state[c[1]] = value
            if c[1] == self._snake[0][1]:
                self._state[c[0] + 10] = value


    def build_state(self):
        self._state *= 0
        if len(self._snake) == 0:
            return self._state.reshape(-1, 1)
        self.vec_repr(self._collectible, -0.4)
        self.vec_repr(self._bad_collectible, -1.0)
        self.vec_repr(self._snake[1:], 0.6)
        self._state[self._snake[0][1]] = 1.0
        self._state[self._snake[0][0] + 10] = 1.0
        return self._state.reshape(-1, 1).copy()
    

    def move(self, curr_dirIdx):
        curr_dir = self._dirs[curr_dirIdx]
        new_pos = [
                    self._player_pos[0] + curr_dir[0], 
                    self._player_pos[1] + curr_dir[1]
                ]
        return new_pos


    def check_self_collision(self, new_pos):
        if new_pos in self._snake:
            self._reward = self._death_penalty 
            self._done = True


    def check_out_of_bounds(self, new_pos) -> bool:
        if new_pos[0] >= self._SIZE or new_pos[0] < 0 or new_pos[1] >= self._SIZE or new_pos[1] < 0:
            self._reward = self._death_penalty 
            self._done = True
            return True
        return False
    

    def mem_num(self) -> int:
        return 3 if self._reward > self._eat_reward / 2.0 else 1


    def handle_step(self, action_idx):

        new_pos = self.move(action_idx)
        # Check if game end
        self.check_self_collision(new_pos)
        self.check_out_of_bounds(new_pos)
        # if game end init, otherwise try to collect item, add reward if snake gets closer to apple
        if self._done != True:
            done = self.collect_item(new_pos)
            if done == True:
                return 0
            if self._reward == 0:
                self._reward += self._lazy_penality
            return 0
        else:
            self._snake, self._collectible, self._bad_collectible, self._currDir, self._player_pos = init_state()
            return 1

    @property
    def reward(self):
        return self._reward
    
    @property
    def done(self):
        return self._done

    def collect_item(self, new_pos) -> bool: # return if snake dead
        self._snake.insert(0, new_pos)
        hit_collectible = False
        for i, item in enumerate(self._collectible):
            if new_pos[0] == item[0] and new_pos[1] == item[1]:
                hit_collectible = True
                self._collectible.pop(i)
                self._reward = self._eat_reward
                add_collectible(self._collectible, self._snake, self._bad_collectible, self._SIZE)
                if len(self._snake) > self._session_max_len:
                    self._session_max_len = len(self._snake)
                break
        hit_bad_collectible = False
        for i, item in enumerate(self._bad_collectible):
            if new_pos[0] == item[0] and new_pos[1] == item[1]:
                hit_bad_collectible = True
                self._bad_collectible.pop(i)
                self._reward = self._eat_penality
                add_collectible(self._bad_collectible, self._snake, self._collectible, self._SIZE)
                break
        # if not hit a collectible, remove the last node, to keep snake length
        if hit_collectible == False:
            self._snake.pop()
        if hit_bad_collectible == True:
            if len(self._snake) == 0:
                self._reward = self._death_penalty
                self._done = True
                return True
            else:
                self._snake.pop()
        self._player_pos = new_pos
        return False


    def update_state(self):
        self._lifetime += 1
        self._final_score += self._reward

    
    def render_text_block(self, pos, title:str, content:list):
        interval = 25
        rel_pos = interval
        draw_text(self._screen, self._font2, title, (pos[0] + 1, pos[1] + 1), SnakeGame.color1)
        draw_text(self._screen, self._font2, title, (pos[0], pos[1]), SnakeGame.color2)
        for c in content:
            draw_text(self._screen, self._font1, c, (pos[0], pos[1] + rel_pos), SnakeGame.color2)
            rel_pos += interval



    def render_dir(self, action_idx):
        dir_posX, dir_posY = 700, 550
        size = 50
        interval = size + 2
        colors = [
            SnakeGame.color_button, 
            SnakeGame.color_button, 
            SnakeGame.color_button, 
            SnakeGame.color_button
        ]
        colors[action_idx] = SnakeGame.color2

        pygame.draw.rect(self._screen, SnakeGame.color1, (dir_posX + interval + 2, dir_posY - interval + 2, size, size), border_radius=15)
        pygame.draw.rect(self._screen, SnakeGame.color1, (dir_posX + 2, dir_posY + 2, size, size), border_radius=15)
        pygame.draw.rect(self._screen, SnakeGame.color1, (dir_posX + interval + 2, dir_posY + 2, size, size), border_radius=15)
        pygame.draw.rect(self._screen, SnakeGame.color1, (dir_posX + interval * 2 + 2, dir_posY + 2, size, size), border_radius=15)

        # -----------------------------------------------------------------------------------
        pygame.draw.rect(self._screen, colors[0], 
                            (dir_posX + interval, dir_posY - interval, size, size), 
                             border_radius=15)
        pygame.draw.rect(self._screen, colors[2], 
                             (dir_posX, dir_posY, size, size), 
                             border_radius=15)
        pygame.draw.rect(self._screen, colors[1], 
                            (dir_posX + interval, dir_posY, size, size),
                            border_radius=15)
        pygame.draw.rect(self._screen, colors[3], 
                            (dir_posX + interval * 2, dir_posY, size, size),
                            border_radius=15)


    def render_bar(self, posX, posY, value, max):
        bar_size = 180
        bar_len = value / max * bar_size
        pygame.draw.rect(self._screen, (255, 255, 255), 
                        (posX , posY, bar_size, 3))
        pygame.draw.rect(self._screen, SnakeGame.color2, 
                        (posX , posY, bar_len, 3))


    def render_view(self, posX, posY):
        bar_len = 18
        for i, item in enumerate(self._state[:10]):
            pygame.draw.rect(self._screen, SnakeGame.color_map[round(float(item) * 10)], 
                             (posX + i * bar_len, posY, bar_len, 3))
        for i, item in enumerate(self._state[10:]):
            pygame.draw.rect(self._screen, SnakeGame.color_map[round(float(item) * 10)], 
                             (posX + i * bar_len, posY + 10, bar_len, 3))


    def update_display(self, action_idx, iteration, epsilon, min_explo_rate, mem_len):
        if self._render == True:
            posX, posY = 680, 30

            self._screen.blit(self._background_img, (0, 0))

            duck_img = self._head_img1 if action_idx % 2 == 1 else self._head_img2
            draw_snake(self._screen, self._snake, self._radius, duck_img, self._body_img)
            draw_item(self._screen, self._collectible, self._bad_collectible, self._radius, self._green_img, self._red_img)
            
            pygame.draw.rect(self._screen, SnakeGame.color1, (posX - 30 + 5, posY - 20 + 5, 260, 610), width=0, border_radius=15)
            pygame.draw.rect(self._screen, SnakeGame.color_bg, (posX - 30, posY - 20, 260, 610), width=0, border_radius=15)
            
            draw_text(self._screen, self._font3, "Duck 2 slither", (posX + 2, posY + 2), SnakeGame.color1)
            draw_text(self._screen, self._font3, "Duck 2 slither", (posX, posY), SnakeGame.color2)
            self.render_text_block((posX, posY + 70), "Mode", [
                f"{'Training' if self._train_mode else 'Playing'}"
            ])
            self.render_text_block((posX, posY + 130), "Stage", [
                f"Lifetime  {self._lifetime}", f"Snake Size  {len(self._snake)}",
                f"Instant Reward  {self._reward}", f"Final Score  {self._final_score:.2f}",
                f"View",])

            explo_rate = max(epsilon, min_explo_rate)
            self.render_text_block((posX, posY + 300), "Record", [
                f"Game Session  {iteration}", f"Max Length  {self._max_len}",
                f"Max Score  {self._max_final_score:.2f}",
                f"Memory Pool  {mem_len} / 5000",
                f"Exploration Rate  {explo_rate:.2f}"])
            self.render_view(posX, posY + 280)
            self.render_bar(posX, posY + 370, self._max_len, 60)
            self.render_bar(posX, posY + 420, mem_len, 5000)
            self.render_bar(posX, posY + 445, explo_rate, 1)
            self.render_dir(action_idx)
            pygame.display.flip()
            self._clock.tick(self._tick_time)


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
                self._tick_time = 5
            elif event.key == pygame.K_e:
                self._tick_time = 5000
            return dirIdx
        return None


    def log_info(self, session):
        len_snake = len(self._snake)
        if len_snake > self._max_len:
            self._max_len = len_snake
        snake = ''.join(('<' if i == 0 else '-') for i in range(len_snake))
        ave_len = int(self._total_len / (session + 1))
        return(f"[SNAKE] (max {self._max_len:2d}  avg {ave_len:2d})  {snake}")