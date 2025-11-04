setup:
	bash venv.sh

train-show:
	python game.py true

train-load:
	python game.py true params.json

train-off:
	python game.py false

train-off-load:
	python game.py false params.json