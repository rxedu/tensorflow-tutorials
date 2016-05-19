all: clean lint test docs

clean:
	@rm -rf docs

docs:
	@pdoc --overwrite --html --html-dir docs tensorflow_tutorials

gh-pages:
	@ghp-import -n -p docs/tensorflow_tutorials

lint:
	@python setup.py lint

test:
	@python setup.py test

.PHONY: docs test
