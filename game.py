from game import *


def main():
	try:
		sgame = SnakeGame()
		sgame.loop(True)
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()