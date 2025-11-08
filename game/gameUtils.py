import pygame
import random as rd
from collections import deque


def draw_snake(screen, snake, radius, head_img, body_img):
	for i,node in enumerate(snake):
		posX = (node[0] + 0.5) * 2 * radius
		posY = (node[1] + 0.5) * 2 * radius
		if i == 0:
			screen.blit(head_img, (posX, posY))
		else:
			screen.blit(body_img, (posX, posY))


def draw_item(screen, items, bad_items, radius, green_img, red_img):
	for i,item in enumerate(items):
		posX = (item[0] + 0.5) * 2 * radius
		posY = (item[1] + 0.5) * 2 * radius
		screen.blit(green_img, (posX, posY))
	for i,bad_item in enumerate(bad_items):
		posX = (bad_item[0] + 0.5) * 2 * radius
		posY = (bad_item[1] + 0.5) * 2 * radius
		screen.blit(red_img, (posX, posY))


def draw_text(screen, font, text, pos, color):
	text_surface = font.render(text, True, color)
	text_rect = text_surface.get_rect(topleft=pos)
	screen.blit(text_surface, text_rect)


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


def create_bad_collectible():
	bad_collectible = [
		[8, 8],
	]
	return bad_collectible


def add_collectible(collectible, snake, other_collect, size):
	while True:
		new_col_pos = [rd.randint(0, size - 1), rd.randint(0, size - 1)]
		conflict = False
		if new_col_pos in snake:
			conflict = True
		if new_col_pos in collectible:
			conflict = True
		if new_col_pos in other_collect:
			conflict = True
		if conflict == False:
			collectible.append(new_col_pos)
			break


def init_state():
	snake = create_snake()
	collectible = create_collectible()
	bad_collectible = create_bad_collectible()
	currdir = (0, -1)
	return snake, collectible, bad_collectible, currdir, snake[0]


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