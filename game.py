import pygame
import numpy as np
import random as rd
import time
from mlp import nn, conf

SIZE = 10

def create_matrix(snake, collectible):
	site = np.zeros((SIZE, SIZE))

	for i, node in enumerate(snake):
		if i == 0:
			site[node[0]][node[1]] = 10
		else:
			site[node[0]][node[1]] = 5
	
	for col in collectible:
		site[col[0]][col[1]] = 2
	return site.reshape(-1, 1)


def draw_snake(screen, snake, radius):
	for i,node in enumerate(snake):
		if i == 0:
			pygame.draw.circle(screen, "red", pygame.Vector2(node[0] * 20, node[1] * 20), radius)
		else:
			pygame.draw.circle(screen, "blue", pygame.Vector2(node[0] * 20, node[1] * 20), radius)


def draw_item(screen, items, radius):
	for i,item in enumerate(items):
		pygame.draw.circle(screen, "green", pygame.Vector2(item[0] * 20, item[1] * 20), radius)


def create_snake():
	snake = [
				[5, 5],
				[5, 6],
				[5, 7]
		  	]
	return snake


def create_collectible():
	collectible = [
		[3, 5],
		[6, 8],
	]
	return collectible


def add_collectible(collectible, snake):
	while True:
		x, y = rd.randint(1, SIZE - 1), rd.randint(1,SIZE - 1)
		onsnake = False
		for node in snake:
			if node[0] == x or node[1] == y:
				onsnake = True
		
		if onsnake == False:
			collectible.append((x, y))
			break


def init_state():
	snake = create_snake()
	collectible = create_collectible()
	currdir = (0, -1)
	return snake, collectible, currdir, snake[0]


def loop(running):
	pygame.init()

	dirs  =  [(0, -1), (0, 1), (-1, 0),  (1, 0)]
	ground_size = 20

	snake, collectible, currDir, player_pos = init_state()
	dirIdx = 0
	radius = 10
	screen = pygame.display.set_mode((ground_size * SIZE, ground_size * SIZE))
	clock = pygame.time.Clock()
	running = True
	hit = False
	tick_time = 10
	score = 0
	
	site_state = create_matrix(snake, collectible)

	states = [site_state]
	scores = [0]

	print(site_state.reshape((SIZE, SIZE)))

	while running:
		hit = False
		screen.fill("yellow")
		score = -1
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_w: #and player_pos[1] - 1 >= 0: 
					dirIdx = 0 # dirUp
				elif event.key == pygame.K_s: #and player_pos[1] + 1 <= ground_size:
					dirIdx = 1 # dirDown
				elif event.key == pygame.K_a: # and player_pos[0] - 1 >= 0:
					dirIdx = 2 # dirLeft
				elif event.key == pygame.K_d: # and player_pos[0] + 1 <= ground_size:
					dirIdx = 3 # dirRight

		# print(site_state.shape)
		# predict_dir = nn.inference(site_state.T)
		# print(predict_dir)
		site_state = create_matrix(snake, collectible)

		predict_dir_idx = nn.inference(site_state.T).argmax(axis=1, keepdims=True)[0][0]
		print(predict_dir_idx)
		currDir = dirs[predict_dir_idx]


		new_pos = [player_pos[0] + currDir[0], player_pos[1] + currDir[1]]

		if new_pos[0] != player_pos[0] or new_pos[1] != player_pos[1]:
			player_pos = new_pos
			for i, node in enumerate(snake):
				if new_pos[0] == node[0] and new_pos[1] == node[1]:
					score = -9
					break
			snake.insert(0, new_pos)

			for i, item in enumerate(collectible):
				if new_pos[0] == item[0] and new_pos[1] == item[1]:
					hit = True
					collectible.pop(i)
					score = 10
					add_collectible(collectible, snake)
					break
			if hit == False:
				snake.pop()
		
		if new_pos[0] >= SIZE or new_pos[0] <= 0 or new_pos[1] >= SIZE or new_pos[1] <= 0 :
			score = -9

		if score == -9:
			snake, collectible, currDir, player_pos = init_state()

		states.append(site_state)
		scores.append(11)
		nn.train_batch_rl(np.squeeze(np.array(states), axis=-1), np.array(scores).reshape(-1, 1), 0.00001)
		draw_snake(screen, snake, radius)
		draw_item(screen, collectible, radius)
		print("[SCORE]", score, "[MEM_LEN]", len(states))
		pygame.display.flip()
		clock.tick(tick_time)
	pygame.quit()


def main():
	loop(True)


if __name__ == "__main__":
	main()