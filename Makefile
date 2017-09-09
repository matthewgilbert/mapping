help:
	@echo 'Make for some simple commands        '
	@echo '                                     '
	@echo ' Usage:                              '
	@echo '     make lint    flake8 the codebase'
	@echo '     make test    run unit tests     '

lint:
	flake8 ./mapping

test:
	pytest mapping/tests -v --cov=mapping --cov-report term-missing
