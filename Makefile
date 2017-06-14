help:
	@echo 'Make for some simple commands        '
	@echo '                                     '
	@echo ' Usage:                              '
	@echo '     make lint    flake8 the codebase'
	@echo '     make test    run unit tests     '

lint:
	flake8 ./mapper

test:
	pytest mapper/tests -v --cov=mapper --cov-report term-missing
