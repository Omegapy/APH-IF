# UV Environment Guide for APH-IF

## Overview

The APH-IF (Advanced Parallel HybridRAG - Intelligent Fusion) project uses **UV** as the primary Python package manager for fast, reliable dependency management and virtual environment handling. UV provides superior performance compared to traditional tools like pip and conda, especially on Windows development environments.

## What is UV?

UV is a modern Python package manager and project manager written in Rust that provides:

- **Fast dependency resolution** - Significantly faster than pip
- **Reliable virtual environment management** - Automatic environment creation and activation
- **Lock file support** - Deterministic builds with `uv.lock` files
- **Cross-platform compatibility** - Optimized for Windows, macOS, and Linux
- **Project-based workflows** - Per-service dependency isolation

## Installation

### Windows (Recommended)

```powershell
# Install UV using the official installer
iwr -useb https://astral.sh/uv/install.ps1 | iex
```

### Alternative Installation Methods

```bash
# Using pip
pip install uv

# Using conda
conda install -c conda-forge uv

# Using homebrew (macOS)
brew install uv
```

### Verify Installation

```powershell
uv --version
```

## APH-IF Project Structure

The APH-IF project uses a **monorepo structure** with isolated service environments:

```
APH-IF-Dev/
├── pyproject.toml          # Root project metadata (minimal)
├── uv.lock                 # Root lock file
├── backend/
│   ├── pyproject.toml      # Backend dependencies
│   └── uv.lock             # Backend lock file
├── data_processing/
│   ├── pyproject.toml      # Data processing dependencies
│   └── uv.lock             # Data processing lock file
├── frontend/
│   ├── pyproject.toml      # Frontend dependencies
│   └── uv.lock             # Frontend lock file
└── common/
    ├── pyproject.toml      # Shared utilities
    └── uv.lock             # Common lock file
```

## Core UV Commands

### Environment Management

```powershell
# Sync dependencies (install/update from lock file)
uv sync

# Sync with upgrade (update to latest compatible versions)
uv sync --upgrade

# Create/activate virtual environment automatically
uv run python script.py

# Install project in development mode
uv sync --dev
```

### Dependency Management

```powershell
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev pytest

# Add with version constraints
uv add "fastapi>=0.100.0,<1.0.0"

# Remove a dependency
uv remove package-name

# Update a specific package
uv add package-name --upgrade
```

### Running Commands

```powershell
# Run Python scripts in the virtual environment
uv run python script.py

# Run modules
uv run python -m module.name

# Run with environment variables
$env:DEBUG = "true"
uv run python script.py

# Run tests
uv run pytest

# Run servers
uv run uvicorn app.main:app --reload
```

## Service-Specific Usage

### Backend Service

```powershell
cd backend

# Install dependencies
uv sync

# Add new dependency
uv add fastapi-cors

# Start development server
uv run uvicorn app.main:app --reload --port 8000

# Run tests
uv run pytest
```

### Data Processing Service

```powershell
cd data_processing

# Install ML/AI dependencies
uv sync

# Add new ML dependency
uv add langchain-experimental

# Run complete pipeline
uv run python -m processing.launch_data_processing --test-mode

# Run specific processing step
uv run python -m processing.run_initial_graph_build

# Start FastAPI service
uv run uvicorn processing.main:app --reload --port 8010
```

### Frontend Service

```powershell
cd frontend

# Install dependencies
uv sync

# Add new UI dependency
uv add plotly

# Start Streamlit app
uv run streamlit run app/bot.py --server.port 8501
```

## Lock Files and Reproducibility

### Understanding Lock Files

Each service has its own `uv.lock` file that ensures reproducible builds:

- **Exact versions** - Pins all dependencies to specific versions
- **Transitive dependencies** - Includes all sub-dependencies
- **Platform-specific** - Handles platform differences
- **Integrity checks** - Verifies package integrity

### Working with Lock Files

```powershell
# Install from lock file (exact versions)
uv sync

# Update lock file with new dependencies
uv add package-name

# Regenerate lock file
uv lock

# Install without updating lock file
uv sync --frozen
```

## Environment Variables and Configuration

### APH-IF Environment Integration

UV works seamlessly with APH-IF's environment management:

```powershell
# Set environment before running
python set_environment.py --mode development
uv run python -m processing.launch_data_processing

# Use environment variables
$env:FORCE_TEST_DB = "true"
$env:MAX_PAGES = "5"
uv run python -m processing.launch_data_processing --test-mode
```

### Development vs Production

```powershell
# Development environment
python set_environment.py --mode development
cd data_processing
uv sync
uv run python -m processing.launch_data_processing --test-mode

# Production environment (use with caution)
python set_environment.py --mode production
cd data_processing
uv sync --frozen  # Use exact lock file versions
uv run python -m processing.launch_data_processing
```

## Best Practices

### 1. Service Isolation

Always work within the appropriate service directory:

```powershell
# ✅ Correct - work in service directory
cd backend
uv add fastapi-cors

# ❌ Incorrect - don't modify root dependencies
uv add fastapi-cors  # from project root
```

### 2. Lock File Management

- **Commit lock files** to version control
- **Use `uv sync`** for consistent environments
- **Use `uv sync --frozen`** in CI/CD pipelines

```powershell
# ✅ Correct - sync from lock file
uv sync

# ✅ Correct - add new dependency (updates lock file)
uv add new-package

# ❌ Incorrect - manual pip install
pip install new-package
```

### 3. Virtual Environment Handling

UV automatically manages virtual environments:

```powershell
# ✅ Correct - UV handles environment automatically
uv run python script.py

# ❌ Unnecessary - manual activation not needed
.venv\Scripts\activate
python script.py
```

### 4. Development Dependencies

Separate development and production dependencies:

```powershell
# Development dependencies
uv add --dev pytest pytest-asyncio black ruff

# Production dependencies
uv add fastapi uvicorn pydantic
```

## Troubleshooting

### Common Issues

#### 1. Lock File Conflicts

```powershell
# If lock file is corrupted or conflicts
rm uv.lock
uv lock
uv sync
```

#### 2. Environment Issues

```powershell
# Clear and recreate environment
rm -rf .venv
uv sync
```

#### 3. Dependency Conflicts

```powershell
# Check dependency tree
uv tree

# Resolve conflicts by updating
uv sync --upgrade
```

#### 4. Windows Path Issues

```powershell
# Ensure UV is in PATH
$env:PATH += ";$HOME\.cargo\bin"

# Or reinstall UV
iwr -useb https://astral.sh/uv/install.ps1 | iex
```

### Performance Optimization

```powershell
# Use UV's parallel installation
uv sync --no-cache  # Skip cache for clean install

# Enable verbose output for debugging
uv sync --verbose

# Use specific Python version
uv sync --python 3.12
```

## Integration with APH-IF Workflows

### Complete Development Setup

```powershell
# 1. Clone repository
git clone https://github.com/Omegapy/APH-IF-Dev.git
cd APH-IF-Dev

# 2. Set environment
python set_environment.py --mode development

# 3. Setup each service
cd backend
uv sync
cd ../data_processing
uv sync
cd ../frontend
uv sync
cd ..

# 4. Run services
# Terminal 1: Backend
cd backend && uv run uvicorn app.main:app --reload --port 8000

# Terminal 2: Data Processing
cd data_processing && uv run uvicorn processing.main:app --reload --port 8010

# Terminal 3: Frontend
cd frontend && uv run streamlit run app/bot.py --server.port 8501
```

### Testing Workflow

```powershell
# Setup test environment
python set_environment.py --mode development --force-test-db true

# Run tests for each service
cd backend && uv run pytest
cd ../data_processing && uv run pytest
cd ../frontend && uv run pytest

# Run integration tests
cd data_processing
$env:FORCE_TEST_DB = "true"
$env:MAX_PAGES = "3"
uv run python -m processing.launch_data_processing --test-mode
```

### Production Deployment

```powershell
# 1. Set production environment
python set_environment.py --mode production

# 2. Install with frozen dependencies
cd backend && uv sync --frozen
cd ../data_processing && uv sync --frozen
cd ../frontend && uv sync --frozen

# 3. Start production services
cd backend && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
cd data_processing && uv run uvicorn processing.main:app --host 0.0.0.0 --port 8010
cd frontend && uv run streamlit run app/bot.py --server.port 8501
```

## Advanced Features

### Custom Scripts

Define custom scripts in `pyproject.toml`:

```toml
[project.scripts]
dev-server = "uvicorn app.main:app --reload"
test-suite = "pytest tests/"
```

```powershell
# Run custom scripts
uv run dev-server
uv run test-suite
```

### Environment-Specific Dependencies

```toml
[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]
prod = ["gunicorn"]
```

```powershell
# Install optional dependencies
uv sync --extra dev
uv sync --extra prod
```

### Python Version Management

```powershell
# Use specific Python version
uv sync --python 3.12

# Check available Python versions
uv python list
```

## Conclusion

UV provides a modern, fast, and reliable package management solution for the APH-IF project. By following these guidelines and best practices, you can:

- Maintain consistent development environments
- Ensure reproducible builds across team members
- Efficiently manage dependencies for multiple services
- Integrate seamlessly with APH-IF's environment management system

For more information, visit the [official UV documentation](https://docs.astral.sh/uv/).
