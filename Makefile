setup:
	bash venv.sh

play:
	python snake.py play params.json

train-show:
	python snake.py show

train-load:
	python snake.py show params.json

train-off:
	python snake.py none

train-off-load:
	python snake.py none params.json