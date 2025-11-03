import pygame
import numpy as np
import random as rd


def create_matrix(snake, collectible, size):
	site = np.zeros((size, size))

	for i, node in enumerate(snake):
		if i == 0:
			site[node[0]][node[1]] = 1
		else:
			site[node[0]][node[1]] = 0.5
	
	for col in collectible:
		site[col[0]][col[1]] = 0.2
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


def add_collectible(collectible, snake, size):
	while True:
		x, y = rd.randint(1, size - 1), rd.randint(1, size - 1)
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