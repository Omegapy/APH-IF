# APH-IF Common Module

Shared configuration and utilities for the APH-IF project services.

## Features

- **Environment-aware Configuration**: Automatic Neo4j instance selection
- **Settings Management**: Centralized configuration with Pydantic
- **Safety Mechanisms**: Environment validation and warnings
- **Logging Utilities**: Shared logging configuration

## Configuration Classes

### `EnvironmentMode`
Enum defining supported environment modes:
- `DEVELOPMENT` - Safe for development work
- `PRODUCTION` - Live data, use with caution
- `TESTING` - Explicit testing scenarios only

### `Settings`
Main configuration class with:
- Automatic Neo4j credential selection
- Environment validation
- Service URL configuration
- API key management

## Usage

```python
from common.config import get_settings

settings = get_settings()

# Get current environment info
neo4j_info = settings.get_current_neo4j_info()
print(f"Using {neo4j_info['instance_type']} database")

# Get Neo4j credentials for current environment
uri, username, password = settings.neo4j.get_neo4j_credentials()
```

## Environment Logic

The configuration automatically selects Neo4j credentials based on:

1. **FORCE_TEST_DB=true** → Uses test database (overrides APP_ENV)
2. **APP_ENV=development** → Uses development database
3. **APP_ENV=production** → Uses production database

## Dependencies

- Pydantic Settings 2.10.1+
- Python Dotenv (for .env file loading)

See `pyproject.toml` for complete dependency list.