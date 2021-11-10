loc:
	wc -l *.py

tests:
	python tests1.py
install:
	sudo apt-get install graphviz graphviz-dev
	pip install pytorch-model-summary
	pip install pygraphviz
