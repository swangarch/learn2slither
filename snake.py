#!/usr/bin/python3

from game import *
import sys


def main():
	try:
		if len(sys.argv)==2 and sys.argv[1] == "show":
			sgame = SnakeGame(render=True)
		elif len(sys.argv)==2 and sys.argv[1] == "none":
			sgame = SnakeGame(render=False)
		elif len(sys.argv)==3:
			if sys.argv[1] == "show":
				sgame = SnakeGame(render=True, weights=sys.argv[2])
			elif sys.argv[1] == "none":
				sgame = SnakeGame(render=False, weights=sys.argv[2])
			elif sys.argv[1] == "play":
				sgame = SnakeGame(render=True, weights=sys.argv[2], train_mode=False, rand_move=False)
		else:
			raise RuntimeError("Failed to initialize game")
		sgame.run(max_iter=50000)
	except Exception as e:
		if sgame and sys.argv[1] != "play":
			sgame.model.save_weights()
			sgame.model.plt.close('all')
		print("Error:", e)


if __name__ == "__main__":
	main()