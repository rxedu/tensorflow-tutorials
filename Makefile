all: lint test

lint:
	@python setup.py lint

test:
	@python setup.py test

.PHONY: docs test
