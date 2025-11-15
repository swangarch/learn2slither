setup:
	bash venv.sh

play:
	python snake.py -t false -l params.json

train-show:
	python snake.py -t true

train-load:
	python snake.py -t true -l params.json

train-off:
	python snake.py -t true -v false

train-off-1:
	python snake.py -t true -v false -i 1 -s ./weights/session1.json

train-off-10:
	python snake.py -t true -v false -i 10 -s ./weights/session10.json

train-off-100:
	python snake.py -t true -v false -i 100 -s ./weights/session100.json

train-off-1000:
	python snake.py -t true -v false -i 1000 -s ./weights/session1000.json

train-off-10000:
	python snake.py -t true -v false -i 10000 -s ./weights/session10000.json

train-off-100000:
	python snake.py -t true -v false -i 100000 -s ./weights/session100000.json

train-off-load:
	python snake.py -t true -v false -l params.json