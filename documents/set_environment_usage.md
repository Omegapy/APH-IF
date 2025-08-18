# Set Environment Module Documentation

## Overview

The `set_environment.py` module is the centralized environment management system for the APH-IF project. It provides safe, controlled switching between development, production, and testing environments with automatic Neo4j database selection and comprehensive configuration management.

## Purpose

This module provides:

- **Environment Mode Management**: Safe switching between development and production environments
- **Database Selection**: Automatic Neo4j instance selection based on environment and test flags
- **Configuration Persistence**: Updates the project's `.env` file for consistent configuration
- **Safety Mechanisms**: Prevents accidental production database usage during development
- **Status Reporting**: Clear visibility into current environment configuration
- **Test Mode Protection**: Isolated test database usage for safe testing

## Architecture

### Core Components

1. **Environment Loading**: Reads configuration from project `.env` file
2. **Credential Selection**: Chooses appropriate Neo4j credentials based on mode and flags
3. **Configuration Updates**: Modifies `.env` file and process environment
4. **Status Display**: Provides clear visibility into current configuration
5. **Safety Validation**: Ensures required credentials are available

### Environment Hierarchy

```
Test Mode (FORCE_TEST_DB=true) → Test Database (highest priority)
       ↓
Production Mode → Production Database
       ↓
Development Mode → Development Database (default)
```

## Configuration

### Environment Modes

#### Development Mode
- **Purpose**: Safe environment for development work
- **Database**: Uses `NEO4J_URI_DEV`, `NEO4J_USERNAME_DEV`, `NEO4J_PASSWORD_DEV`
- **Safety**: Default mode, safe for experimentation
- **Usage**: Day-to-day development and testing

#### Production Mode
- **Purpose**: Live production environment
- **Database**: Uses `NEO4J_URI_PROD`, `NEO4J_USERNAME_PROD`, `NEO4J_PASSWORD_PROD`
- **Safety**: Requires explicit activation, use with extreme caution
- **Usage**: Production deployments only

#### Test Mode
- **Purpose**: Isolated testing environment
- **Database**: Uses `NEO4J_URI_TEST`, `NEO4J_USERNAME_TEST`, `NEO4J_PASSWORD_TEST`
- **Safety**: Overrides mode selection, completely isolated
- **Usage**: Automated testing and safe experimentation

### Command Line Arguments

#### Mode Selection
- `--mode`: Target application mode (`development` | `production`)

#### Test Database Control
- `--force-test-db`: Override to use test database (`true` | `false`)

#### Logging Control
- `--verbose`: Enable/disable verbose logging (`true` | `false`)

#### Status Operations
- `--preview`: Show current status without making changes

### Environment Variables (Set by Module)

#### Core Configuration
- `APP_ENV`: Application environment mode (`development` | `production`)
- `FORCE_TEST_DB`: Test database override flag (`true` | `false`)
- `VERBOSE`: Verbose logging flag (`true` | `false`)

#### Runtime Neo4j Configuration
- `NEO4J_URI`: Active Neo4j connection URI
- `NEO4J_USERNAME`: Active Neo4j username
- `NEO4J_PASSWORD`: Active Neo4j password

## Usage

### Basic Environment Switching

```powershell
# Set development environment (default)
python set_environment.py --mode development

# Set production environment (use with caution)
python set_environment.py --mode production

# Check current environment status
python set_environment.py --preview
```

### Test Database Usage

```powershell
# Enable test database (overrides mode)
python set_environment.py --force-test-db true

# Enable test database with development mode
python set_environment.py --mode development --force-test-db true

# Disable test database
python set_environment.py --force-test-db false
```

### Verbose Logging Control

```powershell
# Enable verbose logging
python set_environment.py --verbose true

# Disable verbose logging
python set_environment.py --verbose false

# Set mode and verbose logging together
python set_environment.py --mode development --verbose true
```

### Status and Preview

```powershell
# Show current environment status
python set_environment.py --preview

# Alternative status check
python set_environment.py --status  # (if supported)
```

## API Reference

### Core Functions

#### apply_settings(mode: str, force_test_db: bool, verbose: bool | None) -> None

```python
def apply_settings(mode: str, force_test_db: bool, verbose: bool | None) -> None:
    """Apply environment mode and DB selection to .env and process env.
    
    Args:
        mode: Environment mode ('development' | 'production')
        force_test_db: Whether to force test database usage
        verbose: Verbose logging setting (None to keep current)
    """
```

**Process:**
1. Loads current `.env` configuration
2. Validates mode parameter
3. Sets core environment flags
4. Selects appropriate Neo4j credentials
5. Updates both `.env` file and process environment

#### _select_credentials(mode: str, force_test_db: bool) -> Tuple[str, str, str]

```python
def _select_credentials(mode: str, force_test_db: bool) -> Tuple[str, str, str]:
    """Select appropriate Neo4j credentials based on mode and test toggle.
    
    Args:
        mode: Environment mode
        force_test_db: Test database override flag
        
    Returns:
        Tuple[str, str, str]: (uri, username, password)
    """
```

**Selection Logic:**
1. If `force_test_db=True`: Use test credentials (highest priority)
2. If `mode="production"`: Use production credentials
3. Otherwise: Use development credentials (default)

#### print_status() -> None

```python
def print_status() -> None:
    """Print current environment status from .env file."""
```

**Status Display:**
- Application environment mode
- Test database flag status
- Verbose logging setting
- Active Neo4j URI and username
- Clear visual formatting

### Utility Functions

#### _load_env() -> None

```python
def _load_env() -> None:
    """Load environment variables from .env file."""
```

#### _set_env_var(key: str, value: str) -> None

```python
def _set_env_var(key: str, value: str) -> None:
    """Set environment variable in both .env file and process environment."""
```

#### _get(key: str, default: str | None = None) -> str | None

```python
def _get(key: str, default: str | None = None) -> str | None:
    """Get environment variable with optional default."""
```

## Environment File Structure

### Required .env Variables

#### Development Credentials
```env
NEO4J_URI_DEV=bolt://localhost:7687
NEO4J_USERNAME_DEV=neo4j
NEO4J_PASSWORD_DEV=development_password
```

#### Production Credentials
```env
NEO4J_URI_PROD=bolt://production-server:7687
NEO4J_USERNAME_PROD=neo4j
NEO4J_PASSWORD_PROD=production_password
```

#### Test Credentials
```env
NEO4J_URI_TEST=bolt://localhost:7688
NEO4J_USERNAME_TEST=neo4j
NEO4J_PASSWORD_TEST=test_password
```

#### Runtime Variables (Set by Module)
```env
APP_ENV=development
FORCE_TEST_DB=false
VERBOSE=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=development_password
```

## Safety Features

### Credential Validation

```python
# Ensures all required credentials are available
if not all([uri, user, pwd]):
    raise SystemExit("Missing Neo4j credentials in .env for the selected mode.")
```

### Mode Validation

```python
# Validates environment mode
if mode not in {"development", "production"}:
    raise SystemExit("--mode must be 'development' or 'production'")
```

### Test Mode Override

```python
# Test mode takes precedence over environment mode
if force_test_db:
    # Use test credentials regardless of mode
    uri = _get("NEO4J_URI_TEST")
    # ...
```

### File Existence Check

```python
# Ensures .env file exists
if not DOTENV_PATH.exists():
    raise FileNotFoundError(f".env not found at {DOTENV_PATH}")
```

## Integration Points

### With Data Processing

```powershell
# Setup environment before data processing
python set_environment.py --mode development --force-test-db true

# Run data processing with configured environment
cd data_processing
uv run python -m processing.launch_data_processing --test-mode
```

### With Backend Services

```powershell
# Configure environment for backend
python set_environment.py --mode development

# Start backend service
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

### With Frontend Services

```powershell
# Setup environment for frontend
python set_environment.py --mode development --verbose true

# Start frontend service
cd frontend
uv run streamlit run app/bot.py --server.port 8501
```

## Best Practices

### Development Workflow

1. **Always Set Environment First**: Configure environment before running any services
2. **Use Test Database for Testing**: Enable `--force-test-db true` for safe testing
3. **Verify Configuration**: Use `--preview` to check current settings
4. **Keep Development as Default**: Use development mode for day-to-day work

### Production Deployment

1. **Explicit Production Mode**: Always explicitly set production mode
2. **Verify Credentials**: Ensure production credentials are correct
3. **Backup Before Changes**: Backup production database before switching
4. **Monitor Environment**: Regularly verify environment configuration

### Testing Procedures

1. **Isolate Test Environment**: Always use test database for automated testing
2. **Clean Test Data**: Regularly clean test database
3. **Verify Test Mode**: Confirm test mode is active before running tests
4. **Reset After Testing**: Return to development mode after testing

## Error Handling

### Missing Credentials

```python
# Error when credentials are missing
SystemExit: Missing Neo4j credentials in .env for the selected mode.

# Solution: Add required credentials to .env file
NEO4J_URI_DEV=bolt://localhost:7687
NEO4J_USERNAME_DEV=neo4j
NEO4J_PASSWORD_DEV=your_password
```

### Invalid Mode

```python
# Error for invalid mode
SystemExit: --mode must be 'development' or 'production'

# Solution: Use valid mode
python set_environment.py --mode development
```

### Missing .env File

```python
# Error when .env file doesn't exist
FileNotFoundError: .env not found at /path/to/.env

# Solution: Create .env file with required variables
```

### Missing Dependencies

```python
# Error when python-dotenv is not installed
SystemExit: python-dotenv is required. Install with: uv add python-dotenv

# Solution: Install dependency
uv add python-dotenv
```

## Examples

### Development Setup

```powershell
# Basic development setup
python set_environment.py --mode development
python set_environment.py --preview

# Development with verbose logging
python set_environment.py --mode development --verbose true
```

### Testing Setup

```powershell
# Safe testing environment
python set_environment.py --mode development --force-test-db true

# Verify test configuration
python set_environment.py --preview

# Run tests
cd data_processing
uv run python -m processing.launch_data_processing --test-mode
```

### Production Setup

```powershell
# Production environment (use with extreme caution)
python set_environment.py --mode production

# Verify production configuration
python set_environment.py --preview

# Ensure production credentials are correct before proceeding
```

### Environment Switching

```powershell
# Switch from development to test
python set_environment.py --force-test-db true

# Switch back to development
python set_environment.py --force-test-db false

# Switch to production (careful!)
python set_environment.py --mode production --force-test-db false
```

## Status Output Example

```
============================================================
APH-IF ENVIRONMENT STATUS (.env)
============================================================
App Environment: DEVELOPMENT
Force Test DB: true
Verbose Logging: true
Neo4j URI: bolt://localhost:7688
Neo4j Username: neo4j
============================================================
```

## Common Workflows

### Daily Development

```powershell
# Start of day - verify environment
python set_environment.py --preview

# If needed, set to development
python set_environment.py --mode development --verbose true
```

### Before Testing

```powershell
# Enable test database
python set_environment.py --force-test-db true

# Verify test mode
python set_environment.py --preview

# Run tests safely
```

### Production Deployment

```powershell
# Backup current environment
python set_environment.py --preview > current_env.txt

# Set production mode
python set_environment.py --mode production

# Verify production configuration
python set_environment.py --preview

# Deploy services
```

This module is essential for safe environment management in the APH-IF project, providing controlled access to different database instances and preventing accidental data corruption through proper environment isolation.
