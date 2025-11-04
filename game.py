from game import *
import sys

def main():
	# try:
	if len(sys.argv)==2 and sys.argv[1] == "true":
		sgame = SnakeGame(render=True)
	elif len(sys.argv)==2 and sys.argv[1] == "false":
		sgame = SnakeGame(render=False)
	elif len(sys.argv)==3:
		if sys.argv[1] == "true":
			sgame = SnakeGame(render=True, weights=sys.argv[2])
		if sys.argv[1] == "false":
			sgame = SnakeGame(render=False, weights=sys.argv[2])

	sgame.loop(True)
	# except Exception as e:
	# 	print("Error:", e)


if __name__ == "__main__":
	main()