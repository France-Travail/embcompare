# Tests
test:
	pytest --cov-report term-missing

# Bandit analysis
bandit:
	bandit -c pyproject.toml -r src