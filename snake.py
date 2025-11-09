#!/usr/bin/python3

from game import *
import sys


def main():
	try:
		dqn, game = None, None
		if len(sys.argv)==2 and sys.argv[1] == "show":
			game = SnakeGame(render=True)
			dqn = DQN(game)
		elif len(sys.argv)==2 and sys.argv[1] == "none":
			game = SnakeGame(render=False)
			dqn = DQN(game)
		elif len(sys.argv)==3:
			if sys.argv[1] == "show":
				game = SnakeGame(render=True)
				dqn = DQN(game, weights=sys.argv[2])
			elif sys.argv[1] == "none":
				game = SnakeGame(render=False)
				dqn = DQN(game, weights=sys.argv[2])
			elif sys.argv[1] == "play":
				game = SnakeGame(render=True)
				dqn = DQN(game, weights=sys.argv[2], train_mode=False)
		else:
			raise RuntimeError("Failed to initialize game")
		dqn.run(max_iter=50000)
	except Exception as e:
		if dqn and sys.argv[1] != "play":
			dqn.model.save_weights()
			dqn.model.plt.close('all')
		print("Error:", e)


if __name__ == "__main__":
	main()