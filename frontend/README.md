# APH-IF Frontend Service

The frontend service provides a Streamlit-based web interface for the Advanced Parallel HybridRAG - Intelligent Fusion system.

## Features

- **Streamlit Web Interface**: Interactive chat interface for natural language queries
- **Environment Awareness**: Visual indicators for current Neo4j environment
- **Real-time Status**: Live environment and database connection status
- **Native Windows Development**: Fast development with uv package manager

## Quick Start

```powershell
# From project root
cd frontend

# Install dependencies
uv sync

# Start the service
uv run streamlit run app/bot.py --server.port 8501
```

## Environment Display

The frontend automatically displays the current environment configuration:

- **🔧 DEVELOPMENT**: Green indicator - safe for development work
- **🚨 PRODUCTION**: Red indicator - live data, use with caution
- **⚠️ TEST**: Yellow indicator - testing database active

The interface shows warnings when connected to production or test databases.

## User Interface

- **Chat Interface**: Natural language query input
- **Environment Status**: Current Neo4j instance and safety warnings
- **Response Display**: APH-IF generated responses with citations
- **Service Health**: Backend and data processing service status

## Development

```powershell
# Run with auto-reload
uv run streamlit run app/bot.py --server.port 8501

# Check environment
python ../check_environment.py
```

## Configuration

The frontend connects to:
- **Backend API**: http://localhost:8000 (configurable via BACKEND_URL)
- **Data Processing**: http://localhost:8010 (for health checks)

## Dependencies

- Streamlit (web framework)
- Requests (HTTP client)
- Pydantic Settings (configuration)

See `pyproject.toml` for complete dependency list.