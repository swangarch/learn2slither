setup:
	bash venv.sh

train-play:
	python game.py play params.json

train-show:
	python game.py show

train-load:
	python game.py show params.json

train-off:
	python game.py none

train-off-load:
	python game.py none params.json