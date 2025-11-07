import pygame
import random as rd
from collections import deque


def draw_snake(screen, snake, radius):
	for i,node in enumerate(snake):
		if i == 0:
			pygame.draw.circle(screen, "red", pygame.Vector2((node[0] + 0.5) * 2 * radius, (node[1] + 0.5) * 2 * radius), radius)
		else:
			pygame.draw.circle(screen, "blue", pygame.Vector2((node[0] + 0.5) * 2 * radius, (node[1] + 0.5) * 2 * radius), radius)


def draw_item(screen, items, radius):
	for i,item in enumerate(items):
		pygame.draw.circle(screen, "green", pygame.Vector2((item[0] + 0.5) * 2 * radius, (item[1] + 0.5) * 2 * radius), radius)


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
		x, y = rd.randint(0, size - 1), rd.randint(0, size - 1)
		onsnake = False
		if [x, y] in snake:
			onsnake = True
		if onsnake == False:
			collectible.append([x, y])
			break


def init_state():
	snake = create_snake()
	collectible = create_collectible()
	currdir = (0, -1)
	return snake, collectible, currdir, snake[0]


def dist(loc1, loc2):
    return ((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)**0.5


def init_mem_pool(memlen_max):
	return [ 
		deque(maxlen=memlen_max),
		deque(maxlen=memlen_max),
		deque(maxlen=memlen_max),
		deque(maxlen=memlen_max),
		deque(maxlen=memlen_max)
	]