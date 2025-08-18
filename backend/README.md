# APH-IF Backend Service

The backend service provides the core REST API for the Advanced Parallel HybridRAG - Intelligent Fusion system, enabling intelligent query processing with parallel retrieval and fusion capabilities.

## Features

- **FastAPI** REST API with automatic OpenAPI documentation
- **Environment-aware Neo4j configuration** with automatic instance selection via `set_environment.py`
- **Health monitoring** with detailed environment and database information
- **Native Windows development** with uv package manager
- **Parallel RAG Processing** with vector and graph retrieval
- **Intelligent Fusion** using LLM-powered result synthesis
- **CORS Support** for frontend integration

## Quick Start

```powershell
# From project root
cd backend

# Install dependencies
uv sync

# Verify environment configuration
python ../check_environment.py

# Start the service
uv run uvicorn app.main:app --reload --port 8000
```

## Environment Management

The backend integrates with the centralized environment management system:

### Environment Modes
- **Development**: Uses `NEO4J_URI_DEV` (safe for development work)
- **Production**: Uses `NEO4J_URI_PROD` (live data - use with extreme caution)
- **Testing**: Uses `NEO4J_URI_TEST` (only when `FORCE_TEST_DB=true`)

### Environment Control

```powershell
# Set development environment
python set_environment.py --mode development

# Enable test database (development only)
python set_environment.py --mode development --force-test-db true

# Check current environment
python set_environment.py --status

# Start service with current environment
uv run uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### Health and Status
- `GET /healthz` - Comprehensive health check with environment information

### Query Processing
- `POST /query` - Main query endpoint for APH-IF parallel retrieval and fusion

### API Documentation
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - ReDoc API documentation

## Development Workflow

### Local Development

```powershell
# 1. Setup development environment
cd backend
python set_environment.py --mode development

# 2. Install dependencies
uv sync

# 3. Start with hot reload
uv run uvicorn app.main:app --reload --port 8000

# 4. Access API documentation
# http://localhost:8000/docs
```

### Testing

```powershell
# Setup test environment
python set_environment.py --mode development --force-test-db true

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_healthz.py

# Run with verbose output
uv run pytest -v

# Cleanup test environment
python set_environment.py --mode development --force-test-db false
```

### Environment Validation

```powershell
# Check environment configuration
python ../check_environment.py

# Validate specific environment
python ../check_environment.py --require-test

# Check service health
curl http://localhost:8000/healthz
```

## UV Package Management

### Dependency Management

```powershell
# Add new dependency
uv add fastapi-cors

# Add development dependency
uv add --dev pytest-asyncio

# Remove dependency
uv remove package-name

# Update dependencies
uv sync --upgrade

# Install from lock file
uv sync
```

### Virtual Environment

```powershell
# Activate virtual environment (if needed)
.venv\Scripts\activate

# Run commands in virtual environment
uv run python script.py
uv run pytest
uv run uvicorn app.main:app --reload
```

## Configuration

### Environment Variables

The backend uses environment-aware configuration through `set_environment.py`:

#### Core Configuration (Managed by set_environment.py)
- `APP_ENV`: Environment mode (development/production)
- `FORCE_TEST_DB`: Test database override
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Active database credentials

#### Service Configuration
- `BACKEND_URL`: Backend service URL (default: http://localhost:8000)
- `DATA_PROCESSING_URL`: Data processing service URL
- `FRONTEND_URL`: Frontend service URL

#### LLM Configuration
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: Primary model for query processing
- `GEMINI_API_KEY`: Gemini API key (optional)

### Settings Class

The backend uses Pydantic settings with automatic environment detection:

```python
from app.config import get_settings

settings = get_settings()
print(f"Environment: {settings.app_env}")
print(f"Database: {settings.get_current_neo4j_info()}")
```

## Dependencies

### Core Dependencies
- **FastAPI 0.115.0+**: Modern web framework
- **Pydantic 2.8.2+**: Data validation and settings
- **Uvicorn**: ASGI server
- **Neo4j Driver**: Database connectivity
- **OpenAI**: LLM integration

### Development Dependencies
- **pytest**: Testing framework
- **pytest-asyncio**: Async testing support
- **httpx**: HTTP client for testing

See `pyproject.toml` and `uv.lock` for complete dependency specifications.

## Integration

### Frontend Integration
- CORS configured for Streamlit frontend
- Health endpoint provides service discovery
- Environment status shared across services

### Data Processing Integration
- Service discovery through health endpoints
- Shared environment configuration
- Coordinated database access

### Common Module Integration
- Shared configuration models
- Logging utilities
- Error handling patterns

## Monitoring and Observability

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/healthz

# Health check with environment info
curl -s http://localhost:8000/healthz | jq .
```

### Logging

```powershell
# Enable verbose logging
$env:VERBOSE = "true"
uv run uvicorn app.main:app --reload --port 8000
```

### Performance Monitoring

```powershell
# Monitor with uvicorn access logs
uv run uvicorn app.main:app --reload --port 8000 --access-log
```

## Troubleshooting

### Common Issues

**Issue**: Service fails to start
```
Solution: Check port availability and environment configuration
netstat -an | findstr :8000
python ../check_environment.py
```

**Issue**: Database connection failed
```
Solution: Verify Neo4j credentials and environment
python set_environment.py --status
```

**Issue**: Wrong environment detected
```
Solution: Set environment correctly using set_environment.py
python set_environment.py --mode development
```

### Error Recovery

```powershell
# Reset environment to development
python set_environment.py --mode development --force-test-db false
python set_environment.py --status

# Restart service
uv run uvicorn app.main:app --reload --port 8000
```

This backend service provides the core API infrastructure for the APH-IF system with comprehensive environment management and safety features.