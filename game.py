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
			if sys.argv[1] == "none":
				sgame = SnakeGame(render=False, weights=sys.argv[2])
			if sys.argv[1] == "play":
				sgame = SnakeGame(render=True, weights=sys.argv[2], randMove=False)

		sgame.loop(True)
	except Exception as e:
		if sgame:
			sgame.model.save_weights()
		print("Error:", e)


if __name__ == "__main__":
	main()