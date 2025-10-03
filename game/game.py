import pygame
import numpy as np


def create_matrix():
	return np.zeros((20, 20))


def draw_snake(screen, snake, radius):
	for i,node in enumerate(snake):
		if i == 0:
			pygame.draw.circle(screen, "red", node, radius)
		else:
			pygame.draw.circle(screen, "blue", node, radius)

def draw_item(screen, items, radius):
	for i,item in enumerate(items):
		pygame.draw.circle(screen, "green", item, radius)


def loop(running):
	pygame.init()

	screen = pygame.display.set_mode((400, 400))

	radius = 10
	dirLeft = pygame.Vector2(-2 * radius, 0)
	dirRight = pygame.Vector2(2 * radius, 0)
	dirUp = pygame.Vector2(0, -2 * radius)
	dirDown = pygame.Vector2(0, 2 * radius)
	
	clock = pygame.time.Clock()
	running = True
	dt = 1
	tick_time = 2
	iteration = 0

	player_pos = pygame.Vector2(200, 200)

	currDir = dirUp
	snake = [
		pygame.Vector2(player_pos.x, player_pos.y),
		pygame.Vector2(player_pos.x, player_pos.y + 2 * radius),
		pygame.Vector2(player_pos.x, player_pos.y + 4 * radius),
		pygame.Vector2(player_pos.x, player_pos.y + 6 * radius),
		pygame.Vector2(player_pos.x, player_pos.y + 8 * radius)
	]

	items = [
		pygame.Vector2(125, 165),
		pygame.Vector2(345, 285),
	]

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		screen.fill("yellow")
		keys = pygame.key.get_pressed()
		if any(keys):
			if player_pos[1] - 2.5 * radius >= 0 and keys[pygame.K_w]:
				currDir = dirUp
			if player_pos[1] + 2.5 * radius <= screen.get_height() and keys[pygame.K_s]:
				currDir = dirDown
			if player_pos[0] - 2.5 * radius >= 0 and keys[pygame.K_a]:
				currDir = dirLeft
			if player_pos[0] + 2.5 * radius <= screen.get_width() and keys[pygame.K_d]:
				currDir = dirRight
		
		new_pos = player_pos + currDir
		if new_pos != player_pos:
			player_pos = new_pos
			snake.insert(0, new_pos)
			snake.pop()
		
		draw_snake(screen, snake, radius)
		draw_item(screen, items, radius)
		pygame.display.flip()
		clock.tick(tick_time)
		iteration += 1
		if tick_time < 10 and iteration % 5 == 0:
			tick_time += 1
	pygame.quit()


def main():
	loop(True)
	

if __name__ == "__main__":
	main()