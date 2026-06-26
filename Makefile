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
	python snake.py -t true -v false -i 1 -s ./models/session1.json

train-off-10:
	python snake.py -t true -v false -i 10 -s ./models/session10.json

train-off-100:
	python snake.py -t true -v false -i 100 -s ./models/session100.json

train-off-1000:
	python snake.py -t true -v false -i 1000 -s ./models/session1000.json

train-off-10000:
	python snake.py -t true -v false -i 10000 -s ./models/session10000.json

train-off-100000:
	python snake.py -t true -v false -i 100000 -s ./models/session100000.json

train-off-load:
	python snake.py -t true -v false -l params.json


.PHONY: setup play train-show train-load train-off \
        train-off-1 train-off-10 train-off-100 train-off-1000 \
        train-off-10000 train-off-100000 train-off-load