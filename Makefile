.PHONY: dev-up dev-down logs test lint type

dev-up:
	docker compose up -d --build

dev-down:
	docker compose down -v

logs:
	docker compose logs -f

test:
	pytest -q

lint:
	ruff check .

type:
	mypy .


