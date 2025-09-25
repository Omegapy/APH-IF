"""Quick Neo4j connectivity check for the APH-IF backend."""

from __future__ import annotations

import logging
import sys
from typing import Any

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

from app.core.config import settings


def _create_driver(uri: str, username: str, password: str) -> Driver:
    """Create a Neo4j driver instance with the provided credentials."""
    return GraphDatabase.driver(uri, auth=(username, password))


def main() -> None:
    """Attempt to connect to Neo4j and run a simple probe query."""
    uri = settings.neo4j_uri
    username = settings.neo4j_username
    password = settings.neo4j_password

    if not uri:
        print("NEO4J_URI is not configured; update backend/.env")
        sys.exit(1)

    if not username or not password:
        print("Neo4j username/password missing; update backend/.env")
        sys.exit(1)

    driver: Driver | None = None
    try:
        driver = _create_driver(uri, username, password)
        with driver.session() as session:
            record: Any = session.run("RETURN 1 AS ok").single()
            value = record["ok"] if record else None
        print(f"Connected to {uri}; test query returned {value}.")
    except Neo4jError as exc:
        logging.exception("Neo4j query failed")
        print(f"Neo4j query failed: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - unexpected failure guard
        logging.exception("Neo4j connection failed")
        print(f"Neo4j connection failed: {exc}")
        sys.exit(1)
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()
